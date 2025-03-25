import torch

torch.set_printoptions(threshold=float('inf'))
data, _ = torch.load("./data/faulty_one_expert_test/test.pt")

new_data = []
print("len data ", len(data))

count = 0
for item in data:
    # Check if the item has indexable structure
    if hasattr(item, "__getitem__"):
        try:
            sequence_len = len(item) // 18

            for i in range(sequence_len):
                new_data.append(item[i*18:(i+1)*18])
                if item[i * 18 + 16] == 4:
                    count += 1
        except (IndexError, TypeError):
            print("Item doesn't have a penultimate element or isn't indexable as expected")
            break


new_data = torch.stack(new_data) if new_data else torch.Tensor([])
print("len new data ", len(new_data))
print(new_data[0:2])
print(f"Count of items with value 4 at penultimate position: {count}")
print(f"Percentage: {(count / (len(data) * sequence_len)) * 100:.2f}%")

torch.save(new_data, "./data/processed_faulty_only_one_expert/test.pt")