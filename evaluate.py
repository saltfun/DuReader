import torch
import args

def evaluate(model, dev_data):
    total, losses = 0.0, []
    device = args.device

    with torch.no_grad():
        model.eval()
        for batch in dev_data:

            input_ids, input_mask,segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device), \
                                     segment_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device))
            loss = loss / args.gradient_accumulation_steps
            losses.append(loss.item())

        for i in losses:
            total += i
        with open("./log", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)


def get_answer_from_start_and_end_position(start, end, q_plus_c):
    if start == 1 and end == 0:
        return ''
    elif start == 2 and end == 1:
        return 'YES'
    elif start == 3 and end == 2:
        return 'NO'
    else:
        return ''.join(q_plus_c[start:end])


def evaluate2(model, data_valid, debug=False):
    true_samples = 0
    false_samples = 0
    device = args.device
    for step in range(len(data_valid) // args.batch_size):
        batch = data_valid[step * args.batch_size: (step + 1) * args.batch_size]

        input_ids = torch.Tensor([feature.input_ids for feature in batch]).long().to(device)
        token_type_ids = torch.Tensor([feature.segment_ids for feature in batch]).long().to(device)
        input_mask = torch.Tensor([feature.input_mask for feature in batch]).long().to(device)
        start_positions = torch.Tensor([feature.start_position for feature in batch]).long().to(device)
        end_positions = torch.Tensor([feature.end_position for feature in batch]).long().to(device)
        only_tokenized_texts = [feature.only_tokenized_text for feature in batch]

        with torch.no_grad():
            model.eval()
            start_position_logits, end_position_logits = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                                               attention_mask=input_mask)

        predicted_start_positions = torch.argmax(start_position_logits, dim=-1, keepdim=False)
        predicted_end_positions = torch.argmax(end_position_logits, dim=-1, keepdim=False)

        for i in range(args.batch_size):
            if predicted_start_positions[i] == start_positions[i] and predicted_end_positions[i] == end_positions[i]:
                true_samples += 1
            else:
                false_samples += 1
                if debug:
                    print(predicted_start_positions[i].item, predicted_end_positions[i].item(),
                          start_positions[i].item(),
                          end_positions[i].item())
                    print('question and context:')
                    print(''.join(only_tokenized_texts[i]))
                    print('predicted answer')
                    print(
                        get_answer_from_start_and_end_position(predicted_start_positions[i], predicted_end_positions[i],
                                                               only_tokenized_texts[i]))
                    print('true answer')
                    print(get_answer_from_start_and_end_position(start_positions[i], end_positions[i],
                                                                 only_tokenized_texts[i]))

    print('evaluation on valid data, exact match: {}/{} = {}'.format(true_samples, (false_samples + true_samples),
                                                                     true_samples / (false_samples + true_samples)))
    return true_samples / (false_samples + true_samples)