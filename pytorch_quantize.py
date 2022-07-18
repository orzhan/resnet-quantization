



model = resnet20().to(device).eval()
model.load_state_dict(torch.load('checkpoints/model-2.th',map_location=torch.device('cpu'))['state_dict'])

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

top1 = validate(val_loader, model, criterion)
print('Evaluation accuracy %2.2f'%(top1))

fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1"]], inplace=True)
for module_name, module in fused_model.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)

model = fused_model
num_calibration_batches = 32

model.eval()
device='cpu'
model.cpu()

qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.qconfig=qconfig
torch.quantization.prepare(model, inplace=True)

validate(train_loader, model, criterion, num_calibration_batches)
torch.quantization.convert(model, inplace=True)

print("Size of model after quantization")
print_size_of_model(model)

top1 = validate(val_loader, model, criterion)
print('Evaluation accuracy %2.2f'%(top1))

