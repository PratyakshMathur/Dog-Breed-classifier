from imports import *
from loaders import *

dog_files = np.array(glob("dogImages/*/*/*"))
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

dog_files_short = dog_files[:100]
VGG16 = models.vgg16(pretrained=True)
use_cuda = torch.cuda.is_available()
if use_cuda:
    VGG16 = VGG16.cuda()


def load_image(img_path):    
    image = Image.open(img_path).convert('RGB')

    in_transform = transforms.Compose([transforms.Resize(size=(244, 244)),transforms.ToTensor()]) 

    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image


def VGG16_predict(img_path):

    img = load_image(img_path)
    if use_cuda:
        img = img.cuda()
    ret = VGG16(img)
    return torch.max(ret,1)[1].item() 


def dog_detector(img_path):
    idx = VGG16_predict(img_path)
    return idx >= 151 and idx <= 268 

### tranfer learning architecture
model_transfer = models.resnet50(pretrained=True)
##freezing the layers
for param in model_transfer.parameters():
    param.requires_grad = False
    
    #classifier layer 
    model_transfer.fc = nn.Linear(2048, 133, bias=True)

if use_cuda:
    model_transfer = model_transfer.cuda()

    
model_transfer.load_state_dict(torch.load('model/model_transfer_dog_breed.pt'))

class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]


def load_input_image(img_path):    
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor(), standard_normalization]) 

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    return image
def predict_breed_transfer(model, class_names, img_path):

    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx]

