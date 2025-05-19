import tenseal as ts
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from model import ConvNet
from datafucker import data_generator

class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

''''
def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = enc_model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)
    
        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1


    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )
'''

def enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride):
    """
    Evaluates the encrypted model on the test set.

    Args:
        context (ts.Context): TenSEAL context.
        enc_model (EncConvNet): The encrypted model.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function (used here for calculating plaintext loss).
        kernel_shape (tuple): Shape of the convolutional kernel (height, width).
        stride (int): Stride of the convolution.
    """
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    total_samples = 0 # Keep track of total samples processed

    print("Starting encrypted evaluation...")

    # Iterate through the test loader batch by batch
    for data, target in test_loader:
        # Iterate through each image in the batch
        for i in range(data.size(0)): # data.size(0) is the batch size
            # Get a single image and its target label
            single_image = data[i].unsqueeze(0) # Add batch dimension back (size 1)
            single_target = target[i].unsqueeze(0) # Add batch dimension back (size 1)

            # Reshape the single image to (height, width) and convert to list for im2col_encoding
            # Assuming input is 1 channel, 28x28
            image_data_list = single_image.view(28, 28).tolist()

            try:
                # Encoding and encryption for the single image
                # ts.im2col_encoding expects: context, input_list, kernel_h, kernel_w, stride
                x_enc, windows_nb = ts.im2col_encoding(
                    context,
                    image_data_list,
                    kernel_shape[0],
                    kernel_shape[1],
                    stride
                )

                # Encrypted evaluation
                enc_output = enc_model(x_enc, windows_nb)

                # Decryption of result
                output = enc_output.decrypt()
                # Convert decrypted list back to a torch tensor, shape (1, num_classes)
                output = torch.tensor(output).view(1, -1)

                # compute loss (using plaintext output and target)
                # Note: Loss calculation is done in plaintext after decryption for evaluation purposes.
                # Homomorphic encryption typically doesn't allow direct loss calculation on ciphertexts.
                loss = criterion(output, single_target)
                test_loss += loss.item()

                # convert output probabilities to predicted class
                _, pred = torch.max(output, 1)

                # compare predictions to true label
                # Use single_target for comparison
                correct = np.squeeze(pred.eq(single_target.data.view_as(pred)))

                # calculate test accuracy for each object class
                label = single_target.item() # Get the scalar label
                class_correct[label] += correct.item()
                class_total[label] += 1
                total_samples += 1 # Increment total sample count

            except Exception as e:
                print(f"Error processing sample {total_samples}: {e}")
                # Optionally skip this sample or handle the error

    # calculate and print avg test loss
    # Divide by total_samples, not sum(class_total), as test_loss accumulates over all samples
    test_loss = test_loss / total_samples
    print(f'Test Loss: {test_loss:.6f}\n')

    # Print accuracy per class
    for label in range(10):
        if class_total[label] > 0: # Avoid division by zero
            print(
                f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
                f'({int(class_correct[label])}/{int(class_total[label])})'
            )
        else:
             print(f'Test Accuracy of {label}: N/A (no samples for this class)')


    # Print overall accuracy
    if total_samples > 0:
        print(
            f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / total_samples)}% '
            f'({int(np.sum(class_correct))}/{int(total_samples)})'
        )
    else:
        print('\nNo samples processed for overall accuracy calculation.')

# Load one element at a time
train_loader, test_loader = data_generator()
# required for encoding
model = ConvNet()
model.load_state_dict(torch.load('plainmodel.pt'))

kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

enc_model = EncConvNet(model)
criterion = torch.nn.CrossEntropyLoss()
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
