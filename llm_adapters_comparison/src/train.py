import torch
import os
import gc

from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, PromptTuningConfig, TaskType, LoraConfig, PrefixTuningConfig
from peft import IA3Config

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from soft_prompt_lora_tiny_bert import SoftPromptLoRAModel, LoRALinear
from prefix_lora_tiny_bert import PrefixLoRAModel, LoRALinear
from transformers import get_scheduler

import data_preparation
import pandas as pd

from torch.nn.utils import clip_grad_norm_
from opacus.accountants.utils import get_noise_multiplier

def training_normal(model,train_loader,optimizer,epochs,scheduler=False):
    num_training_steps = epochs * len(train_loader)
    if scheduler:
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("\nStarting Training...")
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            batch = {k: torch.tensor(v).to(device) 
                if isinstance(v, list) else v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"], 
                            labels=batch["label"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if scheduler:
                lr_scheduler.step()
            optimizer.zero_grad()
        if (epoch) % 1 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
    return model

def training_with_privacy(model,train_loader,
        lr,epochs,epsilon,delta,max_grad_norm,
        scheduler=False,
        weight_decay=0.01
        ):
    # Compute noise multiplier from target ε
    BATCH_SIZE = train_loader.batch_size
    DATASET_SIZE = len(train_loader.dataset)
    sample_rate = BATCH_SIZE / DATASET_SIZE

    NOISE_MULTIPLIER = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant='rdp',
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    if scheduler:
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
            loss = outputs.loss
            loss.backward()

            clip_grad_norm_(model.parameters(), max_grad_norm)

            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=NOISE_MULTIPLIER * max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device,
                    )
                    param.grad += noise

            optimizer.step()
            if scheduler:
                scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model

def model_testing(model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"])
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["label"]).sum().item()
            total += batch["label"].size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def adapter_training_testing(
        adapter_name,
        adapter_config,
        train_loader,
        test_loader,
        number_labels,
        base_model_name="prajjwal1/bert-tiny",
        scheduler=False,
        weight_decay=0
    ):
    
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, 
                                                    num_labels=number_labels)
    match adapter_name:
        case "soft_prompt":
            print("soft_prompt")
            soft_prompt_config = PromptTuningConfig(
                task_type=TaskType.SEQ_CLS,
                num_virtual_tokens=adapter_config['P_Length'],
                tokenizer_name_or_path=base_model_name
            )
            model = get_peft_model(base_model, soft_prompt_config)

        case "prefix":
            print("prefix")
            prefix_config = PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS,
                num_virtual_tokens=adapter_config['P_Length'],
                prefix_projection=False,
            )
            model = get_peft_model(base_model, prefix_config)

        case "lora":
            print("lora")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=adapter_config['r'], 
                lora_alpha=adapter_config['lora_alpha'],
                lora_dropout=adapter_config['lora_dropout'],
            )
            model = get_peft_model(base_model, lora_config)

        case "ia3":
            print("IA^3")
            ia_3_config = IA3Config(
                task_type=TaskType.SEQ_CLS,
                target_modules=["query", "key", "value", "dense"],
                feedforward_modules=["dense"]
            )
            model = get_peft_model(base_model, ia_3_config)

        case "soft_prompt_plus_lora":
            print("soft_prompt_plus_lora")
            model = SoftPromptLoRAModel(
                model_name=base_model_name,
                num_virtual_tokens=adapter_config['P_Length'],
                lora_r=adapter_config['r'],
                lora_alpha=adapter_config['lora_alpha'],
                lora_dropout=adapter_config['lora_dropout'],
                num_labels=number_labels,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # Freeze all model parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze soft prompt parameters
            for param in model.soft_prompt.parameters():
                param.requires_grad = True
            
            # Unfreeze LoRA parameters
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    for param in module.parameters():
                        param.requires_grad = True

        case "prefix_plus_lora":
            print("prefix_plus_lora")
            model = PrefixLoRAModel(
                model_name=base_model_name,
                num_prefix_tokens=adapter_config['P_Length'],
                lora_r=adapter_config['r'],
                lora_alpha=adapter_config['lora_alpha'],
                lora_dropout=adapter_config['lora_dropout'],
                num_labels=number_labels,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # Freeze all model parameters
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze soft prompt parameters
            for param in model.prefix_encoder.parameters():
                param.requires_grad = True
            
            # Unfreeze LoRA parameters
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    for param in module.parameters():
                        param.requires_grad = True

        case "single_layer_finetuning":
            print("single_layer_finetuning")
            model = base_model
            for param in model.base_model.parameters():
                param.requires_grad = False

        case "full_finetuning":
            print("full_finetuning")
            model = base_model
            for param in model.base_model.parameters():
                param.requires_grad = True
    
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    if adapter_config["Epsilon"]>0 and adapter_name in ["ia3",
        "single_layer_finetuning","full_finetuning"]:
        model = training_with_privacy(
            model=model,
            train_loader=train_loader,
            lr=adapter_config["LR"],
            epochs=adapter_config["Epochs"],
            epsilon=adapter_config["Epsilon"],
            delta=adapter_config["Delta"],
            max_grad_norm=adapter_config["GRAD"],
            scheduler=scheduler,
            weight_decay=weight_decay
        )
    elif adapter_config["Epsilon"]>0 and adapter_name in ["soft_prompt",
        "prefix","lora","soft_prompt_plus_lora", "prefix_plus_lora"]:
        privacy_engine = PrivacyEngine()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters())
            , lr=adapter_config["LR"],weight_decay=weight_decay)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=adapter_config["Epochs"],
            target_epsilon=adapter_config["Epsilon"],
            target_delta=adapter_config["Delta"],
            max_grad_norm=adapter_config["GRAD"],
        )
        model = training_normal(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=adapter_config["Epochs"],
            scheduler=scheduler
        )
    elif adapter_config["Epsilon"]==0:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
         lr=adapter_config["LR"],weight_decay=weight_decay)
        model = training_normal(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=adapter_config["Epochs"],
            scheduler=scheduler
        )
    else:
        raise ValueError("Invalid adapter type or invalid Epsilon")
    
    # Testing         
    accuracy = model_testing(
        model=model,
        test_loader=test_loader
    )
    
    return model,accuracy,trainable_params

def train_test_all(model_name,
        datasets,hyper_parameter_config,
        output_path,
        adapter_method_list,
        dataset_list,
        scheduler=True,weight_decay=0,file_tag="eps_inf"):

    parameter_scale_df = pd.DataFrame(columns=['adapter','number of parameters'])
    parameter_scale_df['adapter'] = adapter_method_list
    parameter_scale_df.set_index('adapter', inplace=True)

    accuracy_df = pd.DataFrame(columns=['dataset'] + adapter_method_list)
    accuracy_df['dataset'] = dataset_list
    accuracy_df.set_index('dataset', inplace=True)

    for adapter in hyper_parameter_config.keys():
        dataset_params = hyper_parameter_config[adapter]
        print(f"processing {adapter} ...")
        for dataset_name in dataset_params.keys():
            print(f"processing {dataset_name} ...")
            train_loader, test_loader,number_labels\
            = data_preparation.dataset_preparation(
                datasets[dataset_name], 
                dataset_name, 
                batch_size=dataset_params[dataset_name]['BS'],
                model_name=model_name
            )

            _,accuracy,num_params = adapter_training_testing(
                adapter_name=adapter, 
                adapter_config=dataset_params[dataset_name],
                train_loader=train_loader, 
                test_loader=test_loader,
                number_labels=number_labels,
                base_model_name=model_name,
                scheduler=scheduler,
                weight_decay=weight_decay
            )
            
            accuracy_df.loc[dataset_name, adapter] = accuracy
            parameter_scale_df.loc[adapter, 'number of parameters'] = num_params
            accuracy_df.to_excel(os.path.join(output_path, 
                f"accuracy_results_{file_tag}.xlsx"), index=False)
            parameter_scale_df.to_excel(os.path.join(output_path, 
                f"parameter_scale_{file_tag}.xlsx"), index=False)

            ## TODO: (optional) write the model (adapter+dataset) into drive
            torch.cuda.empty_cache()
            gc.collect()

