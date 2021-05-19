"""
Both in Model_vgg and model_train the path to where the respective models are saved need to be changed
in order to use the API effectively.
You would also have to change the resize parameters based on how your model was trained. I used the following

Transfer Learning from VGG model: 240,240,3
Trained Model: 128,128,3

you would need to change these values
"""

import tkinter as tk
import numpy as np
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Defining the the Tkinter Frame using a inheritance to a subclass Application
class Application(tk.Frame):
    #Initialization of the subclass application and superclass tkinter
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()#Organize all the widgets we will create and place them in the parent widget
        self.parent_widget()  #Parent widget

    #Creating the Parent Widget
    def parent_widget(self):

        canvas = tk.Canvas(self, height=800, width=800, bg='grey')
        canvas.pack()
        #Main Frame
        frame1 = tk.Frame(self, bg='lightgrey')
        frame1.place(relwidth=0.99, relheight=0.99, rely=0.005, relx=0.005)

        #Image Label Frame
        self.frame2 = tk.Frame(self, bg='grey')
        self.frame2.place(in_=frame1,anchor='c', relx=0.5,rely=0.3,relheight=0.5,relwidth=0.5)

        #Text Label Frame
        self.frame3 = tk.Frame(self,bg='#C0C0C0')
        self.frame3.place(in_=frame1,anchor='c',relx=0.5,rely=0.73,relheight=0.2,relwidth=0.5)

        #Frame 1:-------------------------------------------------------------------------------------------------------

        #Import Button
        path = tk.Button(frame1, text='Import', command=self.importimg)
        path.place(rely=0.001, relx=0.001)

        #VGG model button
        vgg_model = tk.Button(frame1, text='Transfer Learning Model VGG', command=self.model_vgg)
        vgg_model.place(rely=0.9,relx=0.5)

        #Trained model button
        train_model = tk.Button(frame1, text ='Trained Model', command=self.model_train)
        train_model.place(rely=0.9,relx=0.3)

        #Quit Button
        quit = tk.Button(frame1, text='QUIT', fg="red", command=self.master.destroy)
        quit.place(rely=0.971, relx=0.001)

    def importimg(self): #Import the image and resize and display in tkinter window

        #Get image path
        self.pathlink = fd.askopenfilename(initialdir="/", title='Please Select Images')

        #Import image
        self.img = Image.open(self.pathlink)
        img =self.img

        #Image preporcessing
        img = img.resize((400, 400), Image.ANTIALIAS)

        #Display image in a label inside the tkinter window
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(self.frame2, image=img)
        panel.image = img
        panel.place(x=0,y=0,relwidth=1,relheight=1)

    def model_vgg(self): #use the imported image and predict using the transfer learning model

        #Image preprocessing
        img = self.img.resize((240, 240), Image.ANTIALIAS)
        img = np.asarray(img) / 255
        img1 = img.reshape(1, 240, 240, 3)

        ## Predictions using the VGG transfered learning model
        model_vgg = load_model('D:\Programing and Tech\Python\AI 2\LAB 4\model1')
        predictions = model_vgg.predict(img1)
        str = f'Prediction for Image: \n{predictions[0][0]:.2%} Cat\n{predictions[0][1]:.2%} Dog'

        # Use Label to display the results
        textlabel = tk.Label(self.frame3)
        textlabel.configure(text=str, font=('Arial',25))
        textlabel.place(x=0,y=0,relwidth=1,relheight=1)

    def model_train(self): #use the imported image and predict using the trained model

        #Image preprocessing
        img = self.img.resize((128, 128), Image.ANTIALIAS)
        img = np.asarray(img) / 255
        img2 = img.reshape(1, 128, 128, 3)

        ## Predictions using the Trained Model
        model_train = load_model('D:\Programing and Tech\Python\AI 2\LAB 4\modelt1')
        predictions = model_train.predict(img2)
        str = f'Prediction for Image: {predictions[0][0]:.0f}'
        if predictions[0][0]<=0.5:
            str = str+'\nIts a Cat'
        else:
            str = str+'\nIts a Dog'
        #Use Label to display the results
        textlabel = tk.Label(self.frame3)
        textlabel.configure(text=str, font=('Arial',25))
        textlabel.place(x=0, y=0, relwidth=1, relheight=1)


root = tk.Tk() #Superclass from tkinter
app = Application(master=root) #inintialize the application
app.mainloop()
