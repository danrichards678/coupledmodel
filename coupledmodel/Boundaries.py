#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:02:21 2021

@author: daniel
"""
from fenics import *
from mshr import *
import numpy as np



class UnitSquare:
    def __init__(self,meshsize):
        self.mesh = UnitSquareMesh(meshsize,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) )

        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()

    def LidDrivenCavity(self,W):
        bc_top  = DirichletBC(W.sub(0), Constant((1.,0)),  self.top)
        bc_bottom   = DirichletBC(W.sub(0), Constant((0,0)),   self.bottom)
        bc_left = DirichletBC(W.sub(0), Constant((0,0)), self.left)
        bc_right = DirichletBC(W.sub(0), Constant((0,0)), self.right)
        self.bcs = [bc_top, bc_bottom, bc_left, bc_right]
        return self.bcs

    def Couette(self,W):
        bc_top  = DirichletBC(W.sub(0), Constant((1.,0)),  self.top)
        bc_bottom   = DirichletBC(W.sub(0), Constant((0,0)),   self.bottom)
        self.bcs = [bc_top, bc_bottom]
        return self.bcs

    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        self.bcs = bc_left
        return self.bcs
    


class AroundCylinder:
    def __init__(self,meshsize):

        channel = Rectangle(Point(0, 0), Point(2.,1.))
        cx=1.2
        cy=0.5
        r = 0.2
        cylinder = Circle(Point(cx,cy), r)
        domain = channel - cylinder
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) )

        # Class describing the top boundary of the domain 
        class cylinderBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>1.0 and x[0]<1.4  \
                    and x[1]>0.2 and x[1]<0.8 )

        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()
        self.wall = cylinderBoundary()


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_wall  = DirichletBC(W.sub(0), Constant((0,0)),   self.wall)
        bc_top = DirichletBC(W.sub(0).sub(1), Constant(0), self.top)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
        self.bcs = [bc_left,bc_wall]#,bc_top,bc_bottom]
        return self.bcs

    def PressureInlet(self,W):
        bc_left =  DirichletBC(W.sub(1), Constant(1), self.left)
        bc_wall  = DirichletBC(W.sub(0), Constant((0,0)),   self.wall)
        bc_top = DirichletBC(W.sub(0).sub(1), Constant(0), self.top)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
        self.bcs = [bc_left,bc_wall,bc_top,bc_bottom]
        return self.bcs



class ChannelExpansion:
    def __init__(self,meshsize):

        lx = 2.
        channellength=0.2
        channelwidth = 0.2

        channel = Rectangle(Point(0, 0.5-0.5*channelwidth), Point(channellength,0.5+0.5*channelwidth))
        expansion = Rectangle(Point(channellength, 0,), Point(lx,1.))
        domain = channel + expansion
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size




        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], lx) )

        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0))

        # Class describing the top boundary of the domain 
        class leftwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], channellength))


        class bottomwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.5-0.5*channelwidth) \
                    and  x[0]<channellength+0.05)

        class topwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.5+0.5*channelwidth) \
                    and  x[0]<channellength+0.05)          

        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()

        self.leftwall = leftwallBoundary()
        self.topwall = topwallBoundary()
        self.bottomwall = bottomwallBoundary()


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_topwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.topwall)
        bc_bottomwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.bottomwall)
        bc_top = DirichletBC(W.sub(0).sub(1), Constant(0), self.top)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
        self.bcs = [bc_left,bc_leftwall,bc_topwall,bc_bottomwall,bc_top,bc_bottom]
        return self.bcs





class Hshape:
    def __init__(self,meshsize):

        lx = 3.
        cutstartx = 1.
        cutlength = 1.
        channelwidth = 0.2

        channel = Rectangle(Point(0, 0), Point(lx,1.))
        topcutout = Rectangle(Point(cutstartx,0.5+0.5*channelwidth),\
            Point(cutstartx+cutlength,1.))
        bottomcutout = Rectangle(Point(cutstartx,0.),\
            Point(cutstartx+cutlength,0.5-0.5*channelwidth))
        domain = channel - topcutout - bottomcutout
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] <cutstartx+0.2 )

        # Class describing the bottom boundary of the domain
        class bottomrightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] >cutstartx+0.2 )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) and x[0] <cutstartx+0.2 )
        
        class toprightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) and x[0] >cutstartx+0.2 )

        # Class describing the top boundary of the domain 
        class leftwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], cutstartx))

        class rightwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], cutstartx+cutlength))

        class bottomwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.5-0.5*channelwidth) \
                    and x[0]>cutstartx -0.05 and x[0]<cutstartx+cutlength+0.05)

        class topwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.5+0.5*channelwidth) \
                    and x[0]>cutstartx -0.05 and x[0]<cutstartx+cutlength+0.05)          

        self.bottomleft = bottomleftBoundary()
        self.bottomright = bottomrightBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.topleft = topleftBoundary()
        self.topright = toprightBoundary()

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)
        

        self.leftwall = leftwallBoundary()
        self.rightwall = rightwallBoundary()
        self.topwall = topwallBoundary()
        self.bottomwall = bottomwallBoundary()


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_topleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.topleft)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_topright = DirichletBC(W.sub(0).sub(1), Constant(0), self.topright)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_rightwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.rightwall)
        bc_topwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.topwall)
        bc_bottomwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.bottomwall)
        self.bcs = [bc_left,bc_topleft,bc_bottomleft,bc_topright,bc_bottomright,bc_leftwall,bc_rightwall,bc_topwall,bc_bottomwall]
        return self.bcs


    def Free(self,W):
        bc_left = DirichletBC(W.sub(1), Constant(8.),self.left)
        bc_right = DirichletBC(W.sub(1),Constant(0.0),self.right)

        bc_topleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.topleft)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_topright = DirichletBC(W.sub(0).sub(1), Constant(0), self.topright)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_rightwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.rightwall)
        bc_topwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.topwall)
        bc_bottomwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.bottomwall)
        
        self.bcs = [bc_topleft,bc_bottomleft,bc_topright,bc_bottomright,bc_leftwall,bc_rightwall,bc_topwall,bc_bottomwall]
        return self.bcs


class TwoCirclesSym:
    def __init__(self,meshsize,r):

        lx = 2.
        cx = 1
        ly=0.5

        cstart = cx-r
        cend = cx+r
    

        channel = Rectangle(Point(0, 0), Point(lx,ly))
        cutout = Circle(Point(cx,0.),r)

        domain = channel - cutout
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] <cstart+0.1 )

        # Class describing the bottom boundary of the domain
        class bottomrightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] >cend-0.1 )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], ly) )
        

        # Class describing the top boundary of the domain 
        class circleBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>cstart-0.05 and x[0]<cend+0.05  \
                    and x[1]>-0.05 and x[1]<r+0.05 )


     
        self.bottomleft = bottomleftBoundary()
        self.bottomright = bottomrightBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.sym = topBoundary()
       
        

        self.circle = circleBoundary()

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)
        
        



     
        


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_circle  = DirichletBC(W.sub(0), Constant((0,0)),   self.circle)
        self.bcs = [bc_left,bc_circle,bc_sym,bc_bottomleft,bc_bottomright]
        return self.bcs


    def Free(self,W):
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_circle  = DirichletBC(W.sub(0), Constant((0,0)),   self.circle)
        self.bcs = [bc_circle,bc_sym,bc_bottomleft,bc_bottomright]
        return self.bcs




class RoundChannel:
    def __init__(self,meshsize,h=0.25,r1=0.2,r2=0.05,sx=0.333,L=0.4):

    

        channel = Rectangle(Point(0, 0), Point(1.,h))
        leftcircle = Circle(Point(sx+r1,0.),r1)
        rightcircle = Circle(Point(sx+L-r2,r1-r2),r2)
        rect1 = Rectangle(Point(sx+r1,0.),Point(sx+L-r2,r1))
        rect2 = Rectangle(Point(sx+L-r2,0.),Point(sx+L,r1-r2))

        domain = channel - leftcircle -rightcircle - rect1 -rect2
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] <sx+0.01 )

        # Class describing the bottom boundary of the domain
        class bottomrightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] >sx+L-0.01 )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], h) )
        

        # Class describing the wall the domain 
        class wallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>sx-0.02 and x[0]<sx+L+0.02  \
                    and x[1]>-0.001 and x[1]<r1+0.01 )


     
        self.bottomleft = bottomleftBoundary()
        self.bottomright = bottomrightBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.sym = topBoundary()
       
        

        self.circle = wallBoundary()

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)

     
        


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_circle  = DirichletBC(W.sub(0), Constant((0,0)),   self.circle)
        self.bcs = [bc_left,bc_circle,bc_sym,bc_bottomleft,bc_bottomright]
        return self.bcs


    def Free(self,W):
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_circle  = DirichletBC(W.sub(0), Constant((0,0)),   self.circle)
        self.bcs = [bc_circle,bc_sym,bc_bottomleft,bc_bottomright]
        return self.bcs




class TwoCircles:
    def __init__(self,meshsize):

        lx = 3.
        cx = 1.5
        r = 0.4

        cstart = cx-r
        cend = cx+r
    

        channel = Rectangle(Point(0, 0), Point(lx,1.))
        topcutout = Circle(Point(cx,0.),r)
        bottomcutout = Circle(Point(cx,1.),r)
        domain = channel - topcutout - bottomcutout
        self.mesh = generate_mesh(domain,meshsize)

        F_el = FiniteElement("Lagrange", triangle, 1)
        F = FunctionSpace(self.mesh, F_el) 
        f = Function(F)
        self.npoints = f.compute_vertex_values().size


        # Class describing the bottom boundary of the domain
        class bottomleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] <cstart+0.1 )

        # Class describing the bottom boundary of the domain
        class bottomrightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) and x[0] >cend-0.1 )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 1.0) )

        # Class describing the top boundary of the domain 
        class topleftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) and x[0] <cstart+0.1 )
        
        class toprightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 1.0) and x[0] >cend-0.1 )

        # Class describing the top boundary of the domain 
        class lowerCircle(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>cstart-0.05 and x[0]<cend+0.05  \
                    and x[1]>-0.05 and x[1]<r+0.05 )

        class upperCircle(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>cstart-0.05 and x[0]<cend+0.05  \
                    and x[1]>1.0-r-0.05 and x[1]<1.0+0.05 )


     
        self.bottomleft = bottomleftBoundary()
        self.bottomright = bottomrightBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.topleft = topleftBoundary()
        self.topright = toprightBoundary()
        

        self.uppercircle = upperCircle()
        self.lowercircle = lowerCircle()
     
        


    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_topleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.topleft)
        bc_bottomleft = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomleft)
        bc_topright = DirichletBC(W.sub(0).sub(1), Constant(0), self.topright)
        bc_bottomright = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottomright)
      
        bc_uppercircle  = DirichletBC(W.sub(0), Constant((0,0)),   self.uppercircle)
        bc_lowercircle  = DirichletBC(W.sub(0), Constant((0,0)),   self.lowercircle)
        self.bcs = [bc_left,bc_uppercircle,bc_lowercircle,bc_topleft,bc_bottomleft,bc_topright,bc_bottomright]
        return self.bcs



class ISMIPB:
    def __init__(self,L,Nx,Nz):
        x = np.linspace(0,L,Nx)
        H=1.
        self.H=H
        self.L=L
        top = np.zeros(Nx)
        bot = -self.H + 0.5*self.H*np.sin(2*np.pi*x/self.L)

        self.mesh = createStructuredMesh(x,bot,top,50,50)

        # Class describing the bottom boundary of the domain
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0) )
            
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L) )

        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>DOLFIN_EPS and x[0]<L \
                    and x[1]<0.4*H)

        self.left = leftBoundary()
        self.right = rightBoundary()
        self.top = topBoundary()
        self.bottom = bottomBoundary()

    def NoSlip(self,W):
        bc_bottom = DirichletBC(W.sub(0), Constant((0,0)),   self.bottom)
        bc_top = DirichletBC(W.sub(0).sub(1), Constant(0), self.top)

        self.bcs = [bc_bottom]
        return self.bcs





# Map the geometry enclosed by bottomSurf and topSurf to structured mesh
def createStructuredMesh(xvec, bottomsurf, topsurf, Nx, Nz):
    hl = Nx-1 # Number of horizontal layers
    vl = Nz-1 # Number of vertical layers
    # generate mesh on unitsquare
    mesh = UnitSquareMesh(hl, vl)
    
    # Extract x and y coordinates from mesh
    x = mesh.coordinates()[:, 0]
    y = mesh.coordinates()[:, 1]
    x0 = min(xvec)
    x1 = max(xvec)
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1-x0)
    zs = np.interp(xnew, xvec, topsurf)
    zb = np.interp(xnew, xvec, bottomsurf)
    ynew = zb + y*(zs-zb)
    
    xynewcoor = np.array([xnew, ynew]).transpose()
    mesh.coordinates()[:] = xynewcoor
    
    return mesh







