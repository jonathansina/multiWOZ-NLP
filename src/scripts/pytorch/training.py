import tqdm
import torch


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    
    total_loss = 0
    num_batches = 0

    for batch in tqdm.tqdm(train_loader, desc="Training"):
        inputs_ids = batch['encoder_input_ids'].to(device)
        attention_mask = batch['encoder_attention_mask'].to(device)
        decoder_ids = batch['decoder_input_ids'].to(device)

        optimizer.zero_grad()

        response_output = model(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            labels=decoder_ids
        )
        loss = response_output.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_epoch(model, valid_loader, device):
    model.eval()
    
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(valid_loader, desc="Validation"):
            inputs_ids = batch['encoder_input_ids'].to(device)
            attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_ids = batch['decoder_input_ids'].to(device)

            response_output = model(
                input_ids=inputs_ids,
                attention_mask=attention_mask,
                labels=decoder_ids
            )
            loss = response_output.loss
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(model, optimizer, scheduler, train_loader, val_loader, device, num_epochs=3, save=False):
    best_loss = float("inf")
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_epoch(model, val_loader, device)

        print(f"Training   - Loss: {train_loss:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    if save:
        torch.save(model.state_dict(), save)
        
    return model