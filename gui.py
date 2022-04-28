from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


def file_dialog(event):
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    image = Image.open(file_path)
    width, height = image.size
    image = image.resize((int(width * 0.5), int(height * 0.5)))
    test = ImageTk.PhotoImage(image)
    label1 = Label(image=test)
    label1.image = test
    label1.pack()


window = Tk()
load_image_button = Button(
    text='Load background',
    width=25,
    height=5,
    bg='gray',
    fg='black'
)
load_image_button.bind("<Button-1>", file_dialog)
load_image_button.pack()
window.mainloop()
