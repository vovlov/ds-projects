"""Train all PyTorch-dependent models inside Docker container."""

from __future__ import annotations

import json
import sys
import os
import pickle
from pathlib import Path

# Add project paths
sys.path.insert(0, "/workspace/03-ner-service")
sys.path.insert(0, "/workspace/04-graph-fraud-detection")
sys.path.insert(0, "/workspace/05-realtime-anomaly")

ARTIFACTS_DIR = Path("/workspace/artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_ner():
    """Fine-tune a transformer model for Russian NER."""
    print("\n" + "=" * 60)
    print("TRAINING: NER Transformer (Russian)")
    print("=" * 60)

    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from datasets import Dataset
    from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_report
    import numpy as np

    # Labels
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    # Training data — expanded for better fine-tuning
    train_data = [
        {"tokens": ["Владимир", "Путин", "посетил", "Москву", "."],
         "ner_tags": [1, 2, 0, 5, 0]},
        {"tokens": ["Компания", "Яндекс", "открыла", "офис", "в", "Санкт-Петербурге", "."],
         "ner_tags": [0, 3, 0, 0, 0, 5, 0]},
        {"tokens": ["Илон", "Маск", "является", "CEO", "Tesla", "."],
         "ner_tags": [1, 2, 0, 0, 3, 0]},
        {"tokens": ["Сбербанк", "заключил", "соглашение", "с", "Газпромом", "."],
         "ner_tags": [3, 0, 0, 0, 3, 0]},
        {"tokens": ["Мария", "Иванова", "переехала", "из", "Казани", "в", "Новосибирск", "."],
         "ner_tags": [1, 2, 0, 0, 5, 0, 5, 0]},
        {"tokens": ["Президент", "Байден", "выступил", "в", "Вашингтоне", "."],
         "ner_tags": [0, 1, 0, 0, 5, 0]},
        {"tokens": ["Google", "и", "Microsoft", "объединили", "усилия", "."],
         "ner_tags": [3, 0, 3, 0, 0, 0]},
        {"tokens": ["Алексей", "Навальный", "родился", "в", "Бутыни", "."],
         "ner_tags": [1, 2, 0, 0, 5, 0]},
        {"tokens": ["ПАО", "Газпром", "нефть", "базируется", "в", "Петербурге", "."],
         "ner_tags": [3, 4, 4, 0, 0, 5, 0]},
        {"tokens": ["Дмитрий", "Медведев", "посетил", "Екатеринбург", "и", "Челябинск", "."],
         "ner_tags": [1, 2, 0, 5, 0, 5, 0]},
        {"tokens": ["Роснефть", "начала", "разработку", "месторождения", "в", "Сибири", "."],
         "ner_tags": [3, 0, 0, 0, 0, 5, 0]},
        {"tokens": ["Анна", "Петрова", "работает", "в", "Тинькофф", "банке", "."],
         "ner_tags": [1, 2, 0, 0, 3, 4, 0]},
        {"tokens": ["МВД", "России", "провело", "операцию", "в", "Краснодаре", "."],
         "ner_tags": [3, 4, 0, 0, 0, 5, 0]},
        {"tokens": ["Сергей", "Лавров", "встретился", "с", "коллегами", "в", "Пекине", "."],
         "ner_tags": [1, 2, 0, 0, 0, 0, 5, 0]},
        {"tokens": ["Amazon", "открыл", "склад", "в", "Германии", "."],
         "ner_tags": [3, 0, 0, 0, 5, 0]},
    ]

    # Use a small multilingual model
    model_name = "bert-base-multilingual-cased"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize and align labels
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=64,
        )
        labels = []
        for i, label_ids in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_word:
                    aligned.append(label_ids[wid])
                else:
                    # For sub-tokens, use I- version if B- was the label
                    lbl = label_ids[wid]
                    if lbl % 2 == 1:  # B- tag
                        aligned.append(lbl + 1)  # I- tag
                    else:
                        aligned.append(lbl)
                prev_word = wid
            labels.append(aligned)
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_dict({
        "tokens": [d["tokens"] for d in train_data],
        "ner_tags": [d["ner_tags"] for d in train_data],
    })

    tokenized = dataset.map(tokenize_and_align, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="/workspace/ner_output",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Training NER model...")
    trainer.train()

    # Save model
    ner_dir = ARTIFACTS_DIR / "ner"
    ner_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(ner_dir))
    tokenizer.save_pretrained(str(ner_dir))
    print(f"NER model saved to {ner_dir}")

    # Quick eval on training data
    model.eval()
    test_text = "Владимир Путин посетил Москву ."
    tokens = test_text.split()
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    word_ids = inputs.word_ids()
    result_labels = []
    prev = None
    for wid, pred in zip(word_ids, preds):
        if wid is not None and wid != prev:
            result_labels.append(id2label[pred])
        prev = wid
    print(f"Test: {tokens} -> {result_labels}")

    return {"status": "ok", "model_path": str(ner_dir)}


def train_gnn():
    """Train GNN model for fraud detection."""
    print("\n" + "=" * 60)
    print("TRAINING: GNN Fraud Detection")
    print("=" * 60)

    import torch
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score, classification_report

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src"):
            del sys.modules[mod_name]
    sys.path.insert(0, "/workspace/04-graph-fraud-detection")
    from src.data.dataset import generate_synthetic_transactions, get_feature_matrix, get_edge_index

    # Generate larger dataset
    data = generate_synthetic_transactions(n_nodes=1000, n_transactions=5000, fraud_rate=0.08)
    X, y = get_feature_matrix(data)
    edge_index = get_edge_index(data)

    print(f"Nodes: {len(X)}, Edges: {edge_index.shape[1]}, Fraud rate: {y.mean():.2%}")

    # Try importing PyG
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, SAGEConv
        import torch.nn.functional as F

        class GCNModel(torch.nn.Module):
            def __init__(self, in_ch, hid_ch=64):
                super().__init__()
                self.conv1 = GCNConv(in_ch, hid_ch)
                self.conv2 = GCNConv(hid_ch, hid_ch)
                self.lin = torch.nn.Linear(hid_ch, 2)

            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                return self.lin(x)

        # Prepare data
        n = len(y)
        perm = np.random.RandomState(42).permutation(n)
        n_train = int(n * 0.8)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[perm[:n_train]] = True
        test_mask = ~train_mask

        graph_data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(y, dtype=torch.long),
        )
        graph_data.train_mask = torch.tensor(train_mask)
        graph_data.test_mask = torch.tensor(test_mask)

        # Train
        model = GCNModel(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        n_pos = y[train_mask].sum()
        n_neg = train_mask.sum() - n_pos
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

        print("Training GCN...")
        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            out = model(graph_data.x, graph_data.edge_index)
            loss = F.cross_entropy(out[graph_data.train_mask], graph_data.y[graph_data.train_mask], weight=weight)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}/300, Loss: {loss.item():.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(graph_data.x, graph_data.edge_index)
            proba = F.softmax(out, dim=1)[:, 1].numpy()
            pred = out.argmax(dim=1).numpy()

        y_test = y[test_mask]
        y_pred = pred[test_mask]
        y_proba = proba[test_mask]

        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0

        print(f"\nGCN Results: F1={f1:.4f}, AUC={auc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Save model
        gnn_dir = ARTIFACTS_DIR / "gnn"
        gnn_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(gnn_dir / "gcn_model.pt"))

        results = {"f1_score": f1, "roc_auc": auc, "model_type": "GCN", "n_nodes": n, "n_edges": edge_index.shape[1]}
        with open(gnn_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"GNN model saved to {gnn_dir}")
        return results

    except ImportError as e:
        print(f"PyTorch Geometric not available: {e}")
        print("Skipping GNN training")
        return {"status": "skipped", "reason": str(e)}


def train_anomaly_lstm():
    """Train LSTM Autoencoder for anomaly detection."""
    print("\n" + "=" * 60)
    print("TRAINING: LSTM Autoencoder for Anomaly Detection")
    print("=" * 60)

    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    # Reset src module cache for anomaly project
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src"):
            del sys.modules[mod_name]
    sys.path.insert(0, "/workspace/05-realtime-anomaly")
    from src.data.generator import generate_timeseries, to_windows

    # Generate training data (normal only) and test data (with anomalies)
    normal_data = generate_timeseries(n_points=5000, anomaly_rate=0.0, seed=42)
    test_data = generate_timeseries(n_points=2000, anomaly_rate=0.03, seed=99)

    X_train, _ = to_windows(normal_data, window_size=30, stride=10)
    X_test, y_test = to_windows(test_data, window_size=30, stride=10)

    # Normalize
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print(f"Train windows: {X_train_norm.shape}, Test windows: {X_test_norm.shape}")
    print(f"Test anomaly rate: {y_test.mean():.2%}")

    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features=3, hidden_size=32, n_layers=1):
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden_size, n_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
            self.output = nn.Linear(hidden_size, n_features)

        def forward(self, x):
            _, (h, c) = self.encoder(x)
            # Repeat hidden state for each timestep
            dec_input = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
            dec_out, _ = self.decoder(dec_input)
            return self.output(dec_out)

    model = LSTMAutoencoder(n_features=3, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)

    print("Training LSTM Autoencoder...")
    model.train()
    batch_size = 64
    for epoch in range(50):
        perm = torch.randperm(len(X_tensor))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[perm[i:i + batch_size]]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/50, Loss: {total_loss / n_batches:.6f}")

    # Evaluate
    model.eval()
    X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        errors = ((X_test_tensor - reconstructed) ** 2).mean(dim=(1, 2)).numpy()

    # Find threshold using percentile
    threshold = np.percentile(errors, 95)
    predictions = (errors > threshold).astype(int)

    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, errors) if len(np.unique(y_test)) > 1 else 0.0

    print(f"\nLSTM AE Results: F1={f1:.4f}, AUC={auc:.4f}")
    print(f"Threshold: {threshold:.6f}")

    # Save
    lstm_dir = ARTIFACTS_DIR / "lstm_ae"
    lstm_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": float(threshold),
    }, str(lstm_dir / "lstm_ae_model.pt"))

    results = {"f1_score": f1, "roc_auc": auc, "threshold": float(threshold)}
    with open(lstm_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"LSTM AE model saved to {lstm_dir}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DOCKER TRAINING: All PyTorch Models")
    print("=" * 60)

    all_results = {}

    # 1. NER
    try:
        all_results["ner"] = train_ner()
    except Exception as e:
        print(f"NER training failed: {e}")
        all_results["ner"] = {"status": "failed", "error": str(e)}

    # 2. GNN
    try:
        all_results["gnn"] = train_gnn()
    except Exception as e:
        print(f"GNN training failed: {e}")
        all_results["gnn"] = {"status": "failed", "error": str(e)}

    # 3. LSTM Autoencoder
    try:
        all_results["lstm_ae"] = train_anomaly_lstm()
    except Exception as e:
        print(f"LSTM AE training failed: {e}")
        all_results["lstm_ae"] = {"status": "failed", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        if "f1_score" in result:
            print(f"  {name}: F1={result['f1_score']:.4f}, AUC={result.get('roc_auc', 'N/A')}")
        else:
            print(f"  {name}: {result.get('status', 'unknown')}")

    with open(ARTIFACTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")
