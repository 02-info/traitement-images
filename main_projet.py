import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from random import randrange 

#################################### PARIE 1 #######################################


def AfficherImg(img):
 
 plt.axis("off")
 plt.imshow(img ,cmap = "gray", vmin=0,vmax=1)
 plt.imshow(img ,cmap = "gray", interpolation="nearest")
 #plt.imshow(img, cmap = "gray")#palette predefinie pour afficher une image
 plt.show()

################# Ouvrir une image jpg ou bmp on retourne une matrice
def ouvrirImage(chemin):
 img=plt.imread(chemin)
 return img

import matplotlib.pyplot as plt
################ Sauvgarder une image sous forme jpg ou bmp
def saveImage(img):
    plt.imsave("image_1.jpg",img)

y=ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
saveImage(y)
    


#################################### PARIE 2 #######################################



def image_noire(h, l):
    # Créer une matrice h x l avec des valeurs initiales de 0
    matrice_image = [[0] * l for _ in range(h)]
    
    return matrice_image


def image_blanche(h, l):
    # Créer une matrice h x l avec des valeurs initiales de 1
    matrice_image = [[1] * l for _ in range(h)]
    
    return matrice_image

def creerImgBlancNoir(h, l):
    Mbn=[[(i+j)%2 for j in range (l)]for i in range (h)]
    return Mbn


# Q7
 
def negatif(img_negatif):
        # Inverser les valeurs de la matrice
    for i in range(len(img_negatif)):
        for j in range(len(img_negatif[0])):
            if img_negatif[i][j]==0:
                img_negatif[i][j]=1
            else :
                img_negatif[i][j]=0
    return img_negatif 


# Q 8 
#                          print(*** TEST TOUTES LES FONCTIONNES ***)

# execute des fonctions de P1

img = ouvrirImage(r"c:\Users\PC\Downloads\png_to.ico")
AfficherImg(img)
#saveImage(chemin)

# EXEMPLE DE FCT 1 P2
hauteur = 3
largeur = 3
matrice_image_noire= image_noire(hauteur, largeur)
# Afficher la matrice (facultatif)
print( "la matrice de l'image noir est : ")
for ligne in matrice_image_noire:
   print(ligne)

# Afficher l'image (facultatif)
#AfficherImg(matrice_image_noire)
    
print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")
 
# EXEMPLE DE FCT2 P2
hauteur = 3
largeur = 3
matrice_image_blanche= image_blanche(hauteur, largeur)

# Afficher la matrice (facultatif)
print( "la matrice de l'image blanche est : ")
for ligne in matrice_image_blanche:
    print(ligne)

#AfficherImg(matrice_image_blanche)

print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")

# EXEMPLE DE FCT3 P2
hauteur = 5
largeur = 5
matrice_image_noire_blanc=creerImgBlancNoir(hauteur, largeur)
AfficherImg(matrice_image_noire_blanc)


# Exemple d'utilisation de la fonction negatif 
img=negatif(matrice_image_noire_blanc)
AfficherImg(img)


#################################### PARIE 3 #######################################



# Q 9 

def luminance(img):
    somme = 0 
    for i in range(len(img)):
        for j in range(len(img[0])):
            # calculer la somme de tous les pixels 
            somme += img[i][j]
    return somme / (len(img) * len(img[0]))
# tset la fonction  
img=ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
reslt_luminance = luminance(img)
print("la luminance de l' image est :",reslt_luminance)

# Q 10 


def contrast(img):
    ctrst = 0
    N = len(img) * len(img[0])
    avg_luminance = luminance(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            ctrst += (img[i][j] - avg_luminance) ** 2
    return ctrst / N

# test the function
img=ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
rslt = contrast(img)
print("le constrast de l' image est : ",rslt )

# Q11 
def profondeur(Img):     
      max= Img[0][0]
      for pixel in Img :
            for ligne in pixel:
                if ligne > max :
                   max = ligne
      return max 
# test the function 
#img=ouvrirImage("Downloads/téléchargement1.jpg")
img=ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
max_pixel = profondeur(img)
print("le profondeur  de l' image est :",max_pixel)

# Q 12
def Ouvrir(Img):
    # plt.imread()  est une fonction dans pyplot qui retourn la matrice d'une image 
    Img=plt.imread(Img)   
    return Img

# test the function 
image_path =(r"c:\Users\PC\Downloads\imagegris1.jpg")
matrice_image = Ouvrir(image_path)
print(matrice_image)



#################################### PARIE 4 #######################################

# Q 13

def inverser(img):
    # cree une matrice initialisé par des zeros    
    invrs=[[0]*len(img[0]) for e in range (len(img))]
    for i  in range (len(img)) :
        for j in range (len(img[0])):
            invrs[i][j]=1- img[i][j]
    return invrs

# test the function 
image_invers=ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
image_invers = inverser(image_invers)
AfficherImg(image_invers)


# Q 14

def flipH(img):
    flip_H =[[0]*len(img[0]) for e in range (len(img))]
    for i  in range (len(img)) :
        for j in range (len(img[0])):
            flip_H[i][j]= img[i][-j]
    return flip_H

# test the function 
image_filp = ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
image_flip = flipH (image_filp)
AfficherImg(image_flip)



# Q 15

def poserV(img1, img2):
    # l'addition de deux images est réalisée élément par élément 
    imgV = img1 + img2
    return imgV

#  TSET THE FUNCTION
image_path = ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
image1=inverser(image_path)
image2=flipH(image_path)
image_poserV=poserV(image1,image2)
AfficherImg(image_poserV)



# Q 16


def poserH(img1, img2): 
    # dans cette liste on parcourir chaque ligne des deux images 
    # et ajouter les éléments correspondants de chaque ligne 
    # pour créer une nouvelle ligne dans la nouvelle image
    imgH = [img1[i] + img2[i] for i in range(len(img1))]
    return imgH

# TEST THE FUNCTION 
image_path = ouvrirImage(r"c:\Users\PC\Downloads\imagegris1.jpg")
image1=inverser(image_path)
image2=flipH(image_path)
image_poserV=poserH(image1,image2)
AfficherImg(image_poserV)


#################################### PARIE 5 #######################################



# Q 22


M = [[[210, 100, 255],[100, 50, 255],[90, 90, 255],[90, 90, 255],[90, 90, 255],[90, 80, 255]],
     [[190, 255, 89],[201, 255, 29],[200, 255, 100],[100, 255, 90],[20, 255, 200],[100, 255, 80]],
     [[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0],[255, 0, 0]]]

# Convert the list to a NumPy array for easier manipulation
M = np.array(M)

plt.imshow(M)
plt.axis ("off")
plt.show ()
# TSET   
print(M[0][1][1])
print(M[1][0][1])
print(M[2][1][0])

# Q 23 

#        *** la response sera sur le rapport *** 

# Q 24 



def initImageRGB(imageRGB):
    # initial chaque pixel de l' image avec  une valeur du couleur aléatoire
    for i in range (len(imageRGB) ) :
        for j in range (len(imageRGB[0]) ):
            imageRGB[i][j]=(randrange(255),randrange(255),randrange(255))
            # renvoyer le tableau image RGB aléatoire 
    return imageRGB 


#  TSET THE FUNCTION
#  créeons d'abord un tableau vide de 3*3 
imageRGB=[[0 for j in range (3) ] for i in range (3)]
imageRGB=initImageRGB(imageRGB)
print(imageRGB)
AfficherImg(imageRGB)

# Q 25 
#            ****** symétrie horizontal ******
Image=ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\tyara.jpeg")
AfficherImg(Image)

def symetrieH(img):
    img_symtrH =[[0]*len(img[0]) for e in range (len(img))]
    for i  in range (len(img)) :
        for j in range (len(img[0])):
            img_symtrH[i][j]= img[i][-j]
    return img_symtrH


# test the function 
image_symtr= ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\tyara.jpeg")
image_symtr = symetrieH(image_symtr)
AfficherImg(image_symtr)

#            ****** symétrie vertical ******

def symetrieV(img):
    img_symtrV =[[0]*len(img[0]) for e in range (len(img))]
    for i  in range (len(img)) :
        for j in range (len(img[0])):
            img_symtrV[i][j]= img[len(img) - 1 - i][j]
    return img_symtrV

# test the function 
image_symtrV= ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\tyara.jpeg")
image_symtrV = symetrieV(image_symtrV)
AfficherImg(image_symtrV)

# Q 26

img_RGB=ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\WhatsApp Image .jpg")
AfficherImg(img_RGB)

def grayscale(imageRGB):
	#### creation d'une novelle matrice 
	img_Gris=[[0]*len(imageRGB[0]) for e in range(len(imageRGB))]
	for i in range(len(imageRGB)):
		for j in range(len(imageRGB[0])):
			elet_max=max(imageRGB[i][j])
			elet_min=min(imageRGB[i][j])
			moy=elet_min # ou moy=elet_min
			img_Gris[i][j]=moy
	return img_Gris 
	
# test the function 
ImgRGB=ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\WhatsApp Image .jpg")
matrice_de_gris=grayscale(ImgRGB)
AfficherImg(matrice_de_gris)
# Q 25  ***  1 / 2 / 3 
ImgRGB=ouvrirImage(r"c:\Users\PC\OneDrive\Bureau\WhatsApp Image .jpg")
listeMax=[]
listeMin=[]
for i in range (len(ImgRGB)):
    for j in range (len(ImgRGB[0])):
        for k in ImgRGB[i][j]:
            max=ImgRGB[i][j][0]
            min=ImgRGB[i][j][0]
            if k>max :
                max=k
            if k <min:
                min=k
        listeMax+=[max]
        listeMin+=[min]

somme1=0
somme2=0
for i in listeMax :
    somme1+=i
for j in listeMin:
    somme2+=j
moyenne1=somme1// len(listeMax)
moyenne2=somme2// len(listeMin)