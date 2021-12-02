
from code import *
from imports import *
def run_app(img_path):

    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    if dog_detector(img_path) is True:
        prediction = predict_breed_transfer(model_transfer, class_names, img_path)
        print("--")
        print("Dogs Detected!\nIt looks like a {0}".format(prediction))  
    elif face_detector(img_path) > 0:
        print("--")
        print("Human Detected!!")
    else:
        print("--")
        print("Error! Can't detect anything..")

for img_file in os.listdir('./Try'):
    img_path = os.path.join('./Try', img_file)
    run_app(img_path)
