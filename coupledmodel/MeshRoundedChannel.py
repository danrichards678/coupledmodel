#%%
from mshr import *
from fenics import *
import numpy as np


def RoundChannel(meshsize,h=0.25,r1=0.225,r2=0.01,sx=0.3,L=0.5):



    channel = Rectangle(Point(0, 0), Point(1.,h))
    leftcircle = Circle(Point(sx+r1,0.),r1)
    rightcircle = Circle(Point(sx+L-r2,r1-r2),r2)
    rect1 = Rectangle(Point(sx+r1,0.),Point(sx+L-r2,r1))
    rect2 = Rectangle(Point(sx+L-r2,0.),Point(sx+L,r1-r2))

    domain = channel - leftcircle -rightcircle - rect1 -rect2
    mesh = generate_mesh(domain,meshsize)
    return mesh


def RoundOutlet(meshsize,h=0.1,r=0.05,L=1.0,cx=0.1):

    outer = Rectangle(Point(0,0), Point(L,0.5))
    cutout1 = Rectangle(Point(0,0), Point(cx-r,0.5-h))
    cutout2 = Rectangle(Point(0,0), Point(cx,0.5-h-r))
    circle = Circle(Point(cx-r,0.5-h-r),r)

    domain = outer - cutout1 - cutout2 -circle

    mesh = generate_mesh(domain,meshsize)
    return mesh
    

def Outlet(meshsize,cy=0.1,L=1.0,cx=0.1):

    
    outer = Rectangle(Point(0,0), Point(L,0.5))
    cutout1 = Rectangle(Point(0,0), Point(cx,0.5-cy))


    domain = outer - cutout1 
    mesh = generate_mesh(domain,meshsize)
    return mesh


meshsize = 25
mesh = RoundOutlet(meshsize)

filename = 'RoundOutlet' + str(meshsize) + '.xml'

File(filename) << mesh


# %%
