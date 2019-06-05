#!/usr/bin/python
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
from skimage.transform import *
from skimage import io
from skimage.transform import resize

import pickle
import numpy as np

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras import optimizers

class VQA_GUI:
    def __init__(self, master):
        self.master = master
        master.title("VQA GUI")

        Label(master,
              text = 'Test the Model',
              fg = "blue",
              font = "Times 10 bold").grid(row = 0, column = 1)
        
        self.q = self.makeentry(master, "Question: ", 1, 0)
        self.pic = self.makeentry(master, "Image Link: ", 2, 0)

        self.photo, height, width = self.get_image("https://static.meijer.com/Media/002/84001/0028400183826_a1c1_0600.png")
        
        self.canvas = Canvas(master, width = width + 10, height = height + 10)
        self.canvas.grid(row = 1, column = 2, columnspan=5, rowspan=5,
               sticky=W+E+N+S, padx=5, pady=5)
        self.image_on_canvas = self.canvas.create_image(0,0, image = self.photo,
                                 anchor = NW)

        self.strQ = StringVar()
        self.labelQ = Label(master, textvariable = self.strQ)
        self.labelQ.grid(row = 3, column = 0)

        self.strClass = StringVar()
        self.labelClass = Label(master, textvariable = self.strClass)
        self.labelClass.grid(row = 4, column = 0)

        self.strPerc = StringVar()
        self.labelPerc = Label(master, textvariable = self.strPerc)
        self.labelPerc.grid(row = 4, column = 1)

        self.greet_button = Button(master, text="Submit", command=self.greet)
        self.greet_button.grid(row = 8, column = 2)

        Label(master,
              text = 'Training the Model',
              fg = "blue",
              font = "Times 10 bold").grid(row = 9, column = 1)
        self.ans = self.makeentry(master, "Correct Answer: ", 10, 0)
        self.ans_button = Button(master, text="Train", command=self.train_save)
        self.ans_button.grid(row = 10, column = 2)

        num_file = open('current_model_num.txt')
        self.counter = int(num_file.readline())
    ###Functions for GUI
    def makeentry(self, parent, caption, r_in, c_in, width=None, **options):
        entry = Entry(parent, **options)
        if width:
            entry.config(width)
        Label(parent, text=caption).grid(row = r_in, column = c_in)
        entry.grid(row = r_in, column = c_in + 1)
        return entry
    def greet(self):
        if self.q.get()!='':
            #print(self.q.get())
            self.trans_q(self.q.get())
            self.output()
            self.q.delete(0, END)
        if self.pic.get()!='':
            self.photo, height, width = self.get_image(self.pic.get())
            self.canvas.itemconfig(self.image_on_canvas, image = self.photo)
            self.pic.delete(0, END)
    def output(self):
        class_perc_dict = self.predict()
        classStr = 'Class:\n'
        percStr = 'Percentage:\n'
        for k in class_perc_dict.keys():
            classStr+=(str(k)+'\n')
            percStr+=(str(class_perc_dict[k])+'%\n')
        self.strQ.set('Current Question: ' + self.q.get())
        self.strClass.set(classStr)
        self.strPerc.set(percStr)
        

    ###Functions for VQA
    def get_model(self):
        json_file = open("model_VQA_4_" + str(self.counter) + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model_VQA_4_" + str(self.counter)+ ".h5")
        return loaded_model
    def predict(self):
        loaded_model = self.get_model()
        # returns an array with percentage posssibility of each class
        perc_arr = loaded_model.predict([self.input_img, self.input_q])[0]

        # returns an array with indices indictating from highest to lowest percentage
        perc_ind = np.array(-perc_arr).argsort()

        my_dict2 = self.get_dict()

        class_perc_dict = dict()
        output = ""
        for i in range(5):
            class_val = my_dict2[perc_ind[i]]
            class_perc = 100 * perc_arr[perc_ind[i]]
            class_perc_dict.update({class_val:class_perc})
        return class_perc_dict
    def trans_q(self, q):
        ## getting the dictionary
        pkl_file = open("word_index_VQA_3.pickle", 'rb')
        word_index = pickle.load(pkl_file)

        words = q.split(' ')
        new_seq = []
        for w in words:
            new_seq.append(word_index.get(w))

        trans_q = np.expand_dims(new_seq, axis = 0)
        self.input_q = pad_sequences(trans_q, maxlen = 50)
    def get_image(self, file_name):
        orig_img = io.imread(file_name)
        new_img_GUI = cv2.resize(orig_img, (224, 224))

        # reshape the image to 224, 224, 3
        reshaped_img = resize(orig_img[:,:,:3], (224,224,3))
        self.input_img = np.expand_dims(reshaped_img, axis=0)

        #self.input_img = np.expand_dims(new_img, axis=0)
        
        height, width, no_channels = new_img_GUI.shape
        photoimg = PIL.Image.fromarray(new_img_GUI)
        photo = PIL.ImageTk.PhotoImage(image = photoimg)
        #self.photo = photo
        #self.photo_height = height
        #self.photo_width = width
        return photo, height, width
    def get_dict(self):
        ## getting ANS_CLASS
        file = open("ans_index_VQA_4.pickle", "rb")
        ans_class = pickle.load(file)
        file.close()

        self.ans_class = ans_class
        my_dict2 = {y:x for x,y in ans_class.items()}
        return my_dict2
    
    def trans_ans(self):
        ### converting the answer to a value
        ans_str = self.ans.get()
        ans_vector = np.zeros(len(self.ans_class))
        ans_vector[self.ans_class[ans_str]] = 1
        ans_vector= np.expand_dims(ans_vector, axis=0)
        self.input_ans = ans_vector
    def train_save(self):
        loaded_model = self.get_model()
        self.trans_ans()
        
        loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
        loaded_model.fit([self.input_img, self.input_q], self.input_ans, epochs=1)

        ## Saving the model
        # serialize model to JSON
        model_json = loaded_model.to_json()
        with open("model_VQA_4_" + str(self.counter + 1) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        loaded_model.save_weights("model_VQA_4_" + str(self.counter + 1) + ".h5")

        self.counter += 1

        num_file = open('current_model_num.txt', 'w')
        num_file.write(str(self.counter))
        print("Saved model to disk")

        
root = Tk()
my_gui = VQA_GUI(root)
root.mainloop()
