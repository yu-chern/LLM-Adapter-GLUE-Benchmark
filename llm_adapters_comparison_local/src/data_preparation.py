import utils
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_tokenize_function(task,model_name="prajjwal1/bert-tiny",TOKEN_MAX_LENGTH=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task == "sst2":
        return lambda examples: tokenizer(examples["sentence"],
                                          truncation=True,
                                          padding="max_length",
                                          max_length=TOKEN_MAX_LENGTH)
    elif task == "qnli":
        return lambda examples: tokenizer(examples["question"],
                                          examples["sentence"],
                                          truncation=True,
                                          padding="max_length",
                                          max_length=TOKEN_MAX_LENGTH)
    elif task == "mnli":
        return lambda examples: tokenizer(examples["premise"],
                                          examples["hypothesis"],
                                          truncation=True,
                                          padding="max_length",
                                          max_length=TOKEN_MAX_LENGTH)
    elif task == "qqp":
        return lambda examples: tokenizer(examples["question1"],
                                          examples["question2"],
                                          truncation=True,
                                          padding="max_length",
                                          max_length=TOKEN_MAX_LENGTH)
    else:
        raise ValueError(f"Unsupported task: {task}")

def dataset_preparation(datasets, dataset_name, batch_size,
                        model_name="prajjwal1/bert-tiny", 
                        token_max_length=128, seed=42):
    utils.set_all_seeds(seed)
    tokenize_function = get_tokenize_function(
        task=dataset_name,
        model_name=model_name,
        TOKEN_MAX_LENGTH=token_max_length
    )
    # Train
    train_dataset = datasets["train"].shuffle(seed=seed)
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_train = tokenized_train.with_format("torch")
    tokenized_train = tokenized_train.remove_columns(
        [col for col in datasets["train"].column_names if col not in ["label"]]
    )
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)

    # Validation
    validation_name = "validation"
    if dataset_name == "mnli":
        validation_name = "validation_matched"
    test_dataset = datasets[validation_name]
    number_labels = len(set(test_dataset["label"]))
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test = tokenized_test.with_format("torch")
    tokenized_test = tokenized_test.remove_columns(
        [col for col in datasets[validation_name].column_names if col not in ["label"]]
    )
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)

    return train_loader, test_loader, number_labels
