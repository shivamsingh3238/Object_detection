from tkinter import *
from tkinter import filedialog
import tkinter.messagebox
from PIL import ImageTk,Image
import cv2
import numpy as np


#----------------------------------------------def----------------
def img():
    a=entry_box.get()
    return a

def src():
    global imgpath
    panel.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    #root.filename.place(x=100,y=80)
    #print(panel.filename)
    
    if (panel.filename)=='':
        entry_box.insert(0,'No Image Selected')
        return (panel.filename)
         
    else:
        tkinter.messagebox.showinfo('Image','Successfully Uploaded !!')
        a=str(panel.filename)
        entry_box.insert(0,a)
    return (panel.filename)
    

#-------------------------------------------sec_screen---------------------
def secscreen():
    panel2 = Toplevel()
    panel2.title("Mini_Project01")
    panel2.geometry("1000x700+120+120")
    top=Frame(panel2,height=150,bg='white')
    top.pack(fill=X)

    bottom=Frame(panel2,height=1000,bg='#42f5bf')
    bottom.pack(fill=X)

    label3=Label(top,text='SHOW',fg='black',bg='yellow',relief='raised',font=("arial",30,"bold"))
    label3.pack(fill=BOTH,pady=4,padx=4)
    b2 = Button(bottom,text="Exit",relief=RAISED,font=("arial",12,'bold'),command = panel2.destroy)
    b2.place(x=1300,y=0)
    label5 = Label(bottom,text="Sourse Image before detection",fg='gray17',bg="white",relief="solid",font=("arial",16,""))
    label5.place(x=90,y=10)
    #-----------image--------
    global contant
    contant=img()
    if contant =='No Image Selected':
        imge=Image.open('C:\Users\$hivam\Desktop\project main\image\\sorry.jpg')
        sizes=(1500,580)
        imge.thumbnail(sizes)
        photo=ImageTk.PhotoImage(imge)
        label6=Label(bottom,image=photo)
        label6.place(x=30,y=40)
    else :
        imge=Image.open(str(contant))
        sizes=(1500,580)
        imge.thumbnail(sizes)
        photo=ImageTk.PhotoImage(imge)
        label6=Label(bottom,image=photo)
        label6.place(x=30,y=40)
    Toplevel.mainloop()

def imgpt():
    a=entry_box.get()
    return a
#-----------------------------------entry box clear---------------------------
def clear():
    entry_box.delete(0,END)
    entry_box.insert(0,'')


#----------------------------------------------------code_file----------------
def run():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    c=imgpt()
    if c=='No Image Selected':
        tkinter.messagebox.showinfo('Image','Frist Upload Image !!')
        c=clear()
        exit
    else:
        img = cv2.imread(str(c))
        print(img.shape)
        img = cv2.resize(img, None, fx=2, fy=0.9)
        height, width, channels = img.shape

        #-----------detect_obj-----
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

    #------------show_img----------
    # Showing informations on the screen

        class_ids = []
        confidences = []
        boxes = []
    
        for out in outs:
    
            for detection in out:
        
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
        
                if confidence > 0.5:
            
            
                # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    #--------------index-------------

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        #cv2.imshow("Image", img)
        height=(1100,770)
        output=cv2.resize(img,height)
        cv2.imshow("Image", output)
        c=clear()
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

 



    
panel = Tk()
contant=''
panel.geometry("1000x700+120+120")
panel.title("Mini_project")
#--------make fram--------
top=Frame(panel,height=150,bg='white')
top.pack(fill=X)

bottom=Frame(panel,height=800,bg='#62dedc')
bottom.pack(fill=X)

 #-----------top fram design----------
label1 = Label(top,text="OBJECT DETECTION",fg='#05ff9b',bg='dark violet',relief="solid",font=("Algerian",55,"bold"))
label1.pack(fill=BOTH,pady=4,padx=4)
label2 = Label(bottom,text="Selected Image Information",fg='gray17',bg="white",relief="solid",font=("Corbel Light",16,"bold")).place(x=717,y=321)
b1 = Button(bottom,text = "Upload Image",activebackground='#eaff05',relief=RAISED,font=("Centaur",30,'bold'),command=lambda :src())
b1.place(x=400,y=325)
#--------entry_box---------
entry_box=Entry(font='Candara 14 bold', width=40,bd=6)
entry_box.insert(0,'')
entry_box.place(x=681,y=450)
#--------------------main_screen_image------------
imge=Image.open('C:\Users\$hivam\Desktop\project main\image\\img.png')
sizes=(1500,580)
imge.thumbnail(sizes)
photo=ImageTk.PhotoImage(imge)
label7=Label(bottom,image=photo)
label7.place(x=300,y=8)
    
b2 = Button(bottom,text="Exit",relief=RAISED,font=("arial",12,'bold'),highlightcolor='#eaff05',command = panel.destroy)
b2.place(x=1300,y=10)
b1 = Button(bottom,text = "Show Uploaded Image",activebackground='#fff305',relief=RAISED,font=("Centaur",30,'bold'),command=secscreen)
b1.place(x=400,y=413)
b1 = Button(bottom,text = "Show Detected Image",activebackground='#ffa305',relief=RAISED,font=("Centaur",30,'bold'),command=run)
b1.place(x=400,y=500)




panel.mainloop()
    
    
    
