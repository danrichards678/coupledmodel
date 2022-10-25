from fenics import *
import numpy as np



class RoundChannel:
    def __init__(self,filename):

        self.mesh = Mesh(filename)

        L = self.mesh.coordinates()[:,0].max()
        h = self.mesh.coordinates()[:,1].max()

        self.L=L
        self.h=h

        #r1=0.2; r2=0.05; sx=0.333; Lc=0.4
        r1 = 0.225; r2=0.01; sx=0.3; Lc=0.5
        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0))

       
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], h) )
        

        # Class describing the wall the domain 
        class wallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>sx-0.02 and x[0]<sx+Lc+0.02  \
                    and x[1]>-0.001 and x[1]<r1+0.01 )


     
        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.sym = topBoundary()
        self.wall = wallBoundary()
        self.pbc = None

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)

    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_wall  = DirichletBC(W.sub(0), Constant((0,0)),   self.wall)
        self.bcs = [bc_left,bc_wall,bc_sym,bc_bottom]
        return self.bcs


    def Free(self,W):
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_wall  = DirichletBC(W.sub(0), Constant((0,0)),   self.wall)
        self.bcs = [bc_wall,bc_sym,bc_bottom]
        return self.bcs

    def TemperatureBC(self,F,Tin=-20):
        bc_leftwall = DirichletBC(F,Constant(Tin), self.left)

        self.Tbcs = [bc_leftwall]
        return self.Tbcs



class RoundOutlet:
    def __init__(self,filename):

        self.mesh = Mesh(filename)

        L = self.mesh.coordinates()[:,0].max()
        h = self.mesh.coordinates()[:,1].max()

        self.L=L
        self.h=h

        r = 0.05
        cw = 0.1
        self.r=0.05
        self.cw =cw
        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0))

       
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], h) )
        

        # Class describing the wall the domain 
        class leftwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>0 and x[0]<2*r+0.02  \
                    and x[1]<h-cw+0.01 and x[1]>0)

        class vertwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0],2*r) and x[1]<h-cw)


        # class circleBoundary(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return bool(on_boundary and x[0]>-0.0 and x[0]<2*r\
        #             x[1]>)

                
                   
        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.sym = topBoundary()
        self.leftwall = leftwallBoundary()
        self.vertwall = vertwallBoundary()
        self.pbc = None

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)

    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_vertwall = DirichletBC(W.sub(0), Constant((0,0)),   self.vertwall)
        self.bcs = [bc_left,bc_leftwall,bc_sym]
        return self.bcs


    def Free(self,W):
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_vertwall = DirichletBC(W.sub(0), Constant((0,0)),   self.vertwall)
        self.bcs = [bc_leftwall,bc_sym]
        return self.bcs

    
    def TemperatureBC(self,F,Tin=-20):
        bc_leftwall = DirichletBC(F,Constant(Tin), self.leftwall)

        self.Tbcs = [bc_leftwall]
        return self.Tbcs



class Outlet:
    def __init__(self,filename):

        self.mesh = Mesh(filename)

        L = self.mesh.coordinates()[:,0].max()
        h = self.mesh.coordinates()[:,1].max()

        self.L=L
        self.h=h

        
        cy = 0.1
        self.cy =cy
        cx = 0.1
        self.cx =0.1
        # Class describing the bottom boundary of the domain
        class bottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], 0.0))

       
        # Class describing the left boundary of the domain
        class leftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], 0.0) )
            
        # Class describing the right boundary of the domain 
        class rightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0], L) )

        # Class describing the top boundary of the domain 
        class topBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[1], h) )
        

        # Class describing the wall the domain 
        class leftwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and x[0]>0 and x[0]<cx+0.01  \
                    and near(x[1],h-cy))

        class vertwallBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool(on_boundary and near(x[0],cx) and x[1]<h-cy+0.01)


        # class circleBoundary(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return bool(on_boundary and x[0]>-0.0 and x[0]<2*r\
        #             x[1]>)

                
                   
        self.bottom = bottomBoundary()
        self.left = leftBoundary()
        self.right = rightBoundary()
        self.sym = topBoundary()
        self.leftwall = leftwallBoundary()
        self.vertwall = vertwallBoundary()
        self.pbc = None

        self.bounds = MeshFunction("size_t", self.mesh ,self.mesh.topology().dim()-1,0)
        self.bounds.set_all(0)
        self.left.mark(self.bounds, 1)

    def ExpressionInlet(self,W,expression,degree=2):
        bc_left =  DirichletBC(W.sub(0), Expression((expression, "0"),degree=degree),self.left)
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_vertwall = DirichletBC(W.sub(0), Constant((0,0)),   self.vertwall)
        self.bcs = [bc_left,bc_leftwall,bc_vertwall,bc_sym]
        return self.bcs


    def Free(self,W):
        bc_sym = DirichletBC(W.sub(0).sub(1), Constant(0), self.sym)
        bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0), self.bottom)
    
        bc_leftwall  = DirichletBC(W.sub(0), Constant((0,0)),   self.leftwall)
        bc_vertwall = DirichletBC(W.sub(0), Constant((0,0)),   self.vertwall)
        self.bcs = [bc_leftwall,bc_vertwall,bc_sym]
        return self.bcs

