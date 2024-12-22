from  scripts.data_loader import get_data_loaders


data_dir = 'dataset'
batch_size = 4

train_loader, test_loader , classes = get_data_loaders(data_dir= data_dir , batch_size=batch_size)

print(f'class to index mapping : {classes}') 

for images , labels in train_loader : 
    print(f'image batch shape : {images.shape}')
    print(f'labels : {labels}')
    break