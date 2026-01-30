# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:40:44 2022

@author: n.patsalidis
"""


import md_pipeline as mdp
import argparse
import numpy as np



def get_stoich(obj,t1,t2,nt1,nt2):
    no,na = nty(obj,t1,t2)
    return na/(na+no)*(nt1+nt2)/nt1

def getNumtype(obj,t):
    if mdp.ass.iterable(t):
        no = 0
        for ti in t:
            no+=np.count_nonzero(obj.at_types==ti)
    else:
        no = np.count_nonzero(obj.at_types==t)
    return no

def nty(obj,t1,t2):

    no = getNumtype(obj,t2)
    na = getNumtype(obj,t1)
    return no,na

def filt_type(obj,ty):
    if mdp.ass.iterable(ty):
        filt = False
        for ti in ty:
            filt = np.logical_or(filt,obj.at_types ==ti )
    else:
        filt = obj.at_types == ty
    return filt

def number_of_atoms_to_remove(obj,t1,t2,nt1,nt2):
    no,na = nty(obj,t1,t2)
    for n in range(no,0,-1):
        xna = na-nt1*n/nt2
        #print(xna)
        if xna%1 ==0 and xna>0:
            return int(xna),int(no-n)
    return

def raise_Exception(var,varname=''):
    if var is None:
        raise Exception('The variable "{}" is None. Give it a proper value'.format(varname))
    return

class shapes():
    def __init__(self,obj,shape,path=None,prefix='nano',**kwargs):
        self.obj = obj # Analysis object
        self.shape = shape
        for k,v in kwargs.items():
            setattr(self,k,v)
        self.cut_shape_func = getattr(self,'cut_'+shape)
        self.get_surface_filt = getattr(self,'surf_'+shape)
        self.get_surfrm_filt = getattr(self,'surfrm_'+shape)
        self.volume = getattr(self,'volume_'+shape)
        self.kwargs = kwargs
        self.name = prefix + self.shape
        if path is None:
            self.path = '{:s}{:s}'.format(prefix,shape)
        else:
            self.path = path
        return

    def get_molname(self):
        x = np.unique(self.obj.mol_names)
        if len(x)>1:
            raise Exception('Found more than one molname in the system')
        else:
            return x[0]

    def cut(self):
        self.cut_shape_func()
        return

    def surf(self,surf_thick):
        filt = self.get_surface_filt(surf_thick)
        return filt

    def surfrm(self,surf_thick):
        filt = self.get_surfrm_filt(surf_thick)
        return filt

    def cut_sphere(self,**kwargs):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords - cm
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fsphere = dcm<=self.diameter/2
        self.obj.filter_system(fsphere)
        return

    def cut_pore(self):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fpore = dcm>=self.diameter/2
        rz = np.abs(coords[:,2]-cm[2])
        fl = rz < self.length/2

        fpore = np.logical_and(fpore,fl)
        self.obj.filter_system(fpore)
        return

    def cut_tube(self):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        finner = dcm>=self.diameter/2
        fouter = dcm<=self.outer_diameter/2
        ftube = np.logical_and(fouter,finner)

        rz = np.abs(coords[:,2]-cm[2])
        fl = rz < self.length/2

        flntube = np.logical_and(ftube,fl)
        self.obj.filter_system(flntube)
        return

    def surf_pore(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fp_s = dcm<=self.diameter/2+surf_thick

        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        s = np.logical_or(fl_s,fp_s)
        return s

    def surf_tube(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords[:,0:2] -cm[0:2]
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fin_s = dcm<=self.diameter/2+surf_thick
        fout_s = dcm>=self.outer_diameter/2-surf_thick
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        s = np.logical_or(fl_s, np.logical_or(fin_s,fout_s))
        return s

    def surf_sphere(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rcm = coords -cm
        dcm = np.sqrt(np.sum(rcm*rcm,axis=1))
        fin_s = dcm>=self.diameter/2-surf_thick
        return fin_s

    def surfrm_pore(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        return fl_s

    def surfrm_tube(self,surf_thick):
        coords = self.obj.get_coords(0)
        cm = mdp.CM(coords,self.obj.atom_mass)
        rz = np.abs(coords[:,2]-cm[2])
        fl_s = rz > self.length/2-surf_thick
        return fl_s

    def surfrm_sphere(self,surf_thick):
        return self.surf_sphere(surf_thick)


    def volume_pore(self):
        vc = 0.25*np.pi*self.length*self.diameter**2
        box = self.obj.get_box(0)
        return box[0]*box[1]*self.length-vc

    def volume_tube(self):
        vc = 0.25*np.pi*self.length*(self.outer_diameter**2 - self.diameter**2)
        return vc

    def volume_sphere(self):
        vc = 4/3*np.pi*self.diameter**3
        return vc

    def surface_pore(self):
        s1 = np.pi*self.length*self.diameter
        box = self.obj.get_box(0)
        s2 = box[0]*box[1]-0.25*np.pi*self.diameter**2
        return s1 + 2*s2

    def surface_tube(self):

        s1 = np.pi*self.length*self.diameter
        s2 = np.pi*self.length*self.outer_diameter
        s3 = 0.25*np.pi*(self.outer_diameter**2-self.diameter**2)
        return s1+s2+2*s3

    def surface_sphere(self):
        s = 4*np.pi*self.diameter**2
        return s


def make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick):
        raise_Exception(type1,'type1')
        raise_Exception(type2,'type2')
        raise_Exception(ntype1,'ntype1')
        raise_Exception(ntype2,'ntype2')
        print('Making the system stoichiometric ... ')
        stoich = get_stoich(obj, type1, type2, ntype1, ntype2)
        print('Cutted shape {:s} stoichiometry = {:4.8f}'.format(shape.name,stoich))

        # Atoms to remove for making stoichiometric
        print('Current non-stoichiometric n{:},n{:} = {:}'.format(type1,type2,nty(obj,type1,type2)))
        na,no = number_of_atoms_to_remove(obj,type1,type2,ntype1,ntype2)

        print('Removing {:d} {:} and {:d}  {:} from the surface'.format(na,type1,no,type2))
        fsurf = shape.surfrm(surf_thick)
        fsurf1 = np.logical_and(fsurf,filt_type(obj,type1))
        fsurf2 = np.logical_and(fsurf,filt_type(obj,type2))
        nsurf1 = np.count_nonzero(fsurf1)
        nsurf2 = np.count_nonzero(fsurf2)
        print('Total surface {:} = {:d}, Total surface {:} = {:d}'.format(type1,nsurf1,type2,nsurf2))
        if na > nsurf1:
            raise Exception('surface {} atoms < {} atoms that need to be removed'.format(type1,type1))
        if no > nsurf2:
            raise Exception('surface {} atoms < {} atoms that need to be removed'.format(type2,type2))

        #Performing the removal
        args1 = obj.at_ids[fsurf1]
        args1 = np.random.choice(args1.copy(),na,replace=False)
        args2 = obj.at_ids[fsurf2]
        args2 = np.random.choice(args2.copy(),no,replace=False)

        args = list(np.concatenate((args1,args2),dtype=int))

        filt = np.ones(obj.natoms,dtype=bool)
        for i in range(obj.natoms):
            if i in args:
                filt[i] = False

        assert  obj.natoms - np.count_nonzero(filt) == args1.shape[0] + args2.shape[0],'number of args to be removed is not equal to False in filter'

        obj.filter_system(filt)
        print('New numbers of natoms {},{} = {}'.format(type1,type2,nty(obj,type1,type2)))
        print('Stoichiometry  of {:s} = {:4.8f} after atom removal'.format(shape.name,get_stoich(obj,type1,type2,ntype1,ntype2)))
        return

def change_surface_types(surface_types_map,obj,shape,surf_thick):
    for key,val in surface_types_map.items():
        fch = np.logical_and(shape.surf(surf_thick),filt_type(obj, key))
        for i in range(obj.natoms):
            if fch[i]:
                obj.at_types[i] = val
        newty_filt = filt_type(obj,val)
        nfch = np.count_nonzero(newty_filt)
        print('Number of changed surface atom types: n{:s} = {:d} , '.format(val,nfch))
    return

def wrap_surface(obj, box_input, diameter=None,length=None,
                    nocut=False,
                    surf_thick=0.3,
                    type1=None, type2=None,
                    ntype1=None, ntype2=None,
                    surface_types_map=None):
    def norm2(r):
        return np.sqrt(np.dot(r,r))

    # find dimensions
    c = obj.get_coords(0)
    thickness = c[:,2].max()-c[:,2].min()
    surf_size = obj.get_coords(0).max(axis=0) - obj.get_coords(0).min(axis=0)
    # find how to multiply
    mult_1 = int((np.pi*diameter+thickness/2)/surf_size[1])
    mult_0 = int(length/surf_size[0])

    multiplicity = (mult_0,mult_1,0)

    obj.multiply_periodic(multiplicity)
    diff = obj.get_coords(0) - obj.get_coords(0).min(axis=0)
    if nocut==False:
        fl = diff[:,0] < length
        fw = diff[:,1] < np.pi*diameter+thickness/2
        obj.filter_system(np.logical_and(fl,fw))
        box = obj.get_coords(0).max(axis=0) - obj.get_coords(0).min(axis=0)
        obj.timeframes[0]['boxsize'] = np.array( box )
    else:
        box = obj.get_box(0)
        


    # wrap_the surface
    coords = obj.get_coords(0)
    thickness = coords[:,2].max() - coords[:,2].min()


    cm = mdp.CM(coords,obj.atom_mass)
    
    coords-=cm
    L = box[1]

    wrapped_coords = np.empty_like(coords)

    cm = mdp.CM(coords,obj.atom_mass)
    rz = cm.copy() ; rz[2] -= L/(4*np.pi)
    d = norm2(rz)
    theta = []
    for i in range(coords.shape[0]):
        rold = coords[i] - rz

        th = (rold[1]-rz[1])*2*np.pi/L
        theta.append(th)

        rr = rold-cm
        rrot = np.array( [rold[0],(d+rr[2])*np.cos(th),(d+rr[2])*np.sin(th)] )

        wrapped_coords[i] = rrot


    wc = wrapped_coords.copy()
    wc[:,0]=wrapped_coords[:,2]
    wc[:,2] = wrapped_coords[:,0]
    wrapped_coords = wc
    wrappedtube_size = wrapped_coords.max(axis=0) - wrapped_coords.min(axis=0)
    
    if nocut == False:
        box = box_input
    else:
        #box = np.array([box_input[0], box_input[1], box[2]])
        
        box = box_input
        nbins = 200
        r = wrapped_coords[:,2]
        hist, edges = np.histogram(r, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # very simple peak picking (works if shells are sharp)
        m = hist.mean() + hist.std()
        peak_idx = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]) & (hist[1:-1] > m))[0] + 1
        shell_r = np.sort(centers[peak_idx])
        dr = np.diff(shell_r).min()   # spacing between successive shells
        box[2] = wrappedtube_size[2]  + dr
        #
        
    obj.timeframes[0]['boxsize'] = box

    alu_cm = mdp.CM(obj.get_coords(0),obj.atom_mass)
    #wrapped_coords+=box/2
    obj.timeframes[0]['coords'] = wrapped_coords

    obj.timeframes[0]['coords'] += obj.get_box(0)/2 -alu_cm

    final_coords = obj.get_coords(0)
    cm = mdp.CM(final_coords,obj.atom_mass)
    fcm = final_coords[:,0:2] - cm[0:2]
    inner_diameter = 2*np.sqrt(np.sum(fcm*fcm,axis=1)).min()
    outer_diameter = 2*np.sqrt(np.sum(fcm*fcm,axis=1)).max()
    length = wrappedtube_size[2]
    for k,s in zip([inner_diameter,outer_diameter,length],['inner_diameter','outer_diameter','length']):
        print('final {:s} = {:4.3f}'.format(s,k))

    if inner_diameter > min(box[0],box[1]):
        raise Exception('Final inner_diameter = {:4.3f} is more than the box size, increase the box size or reduce diameter'.format(inner_diameter))
    elif outer_diameter > min(box[0], box[1]):
        raise Exception('Final outer_diameter = {:4.3f} is more than the box size, increase the box size or reduce diameter'.format(outer_diameter))
    elif length > box[2] and nocut==False:
        raise Exception('Final length = {:4.3f} is more than the box size ({:4.3f}), increase the box size or reduce length'.format(length,box[2]))

    shape = shapes(obj,'tube',prefix='wrapped',
                   diameter=inner_diameter,
                   outer_diameter=outer_diameter,
                   length=length)
    if nocut==False:
        make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick)
    if surface_types_map is not None:
        change_surface_types(surface_types_map,obj,shape,surf_thick)

    obj.mol_ids = np.ones(obj.natoms,dtype=int)

    return obj,shape

def make_nano(obj, nanoshape, box,
                    make_tetr=True,make_stoich=True,type1=None, type2=None,
                    ntype1=None, ntype2=None, surface_types_map = dict(),
                    surf_thick=0.3,  **shape_kwargs):
    
    xdim, ydim, zdim = box
    
    box = obj.get_box(0)

    multiplicity=(int(xdim/box[0]),int(ydim/box[1]),int(zdim/box[2]))

    obj.multiply_periodic(multiplicity)

    coords = obj.get_coords(0)


    fx = np.logical_and(coords[:,0]>=0,coords[:,0]<xdim)
    fy = np.logical_and(coords[:,1]>=0,coords[:,1]<ydim)
    fz = np.logical_and(coords[:,2]>=0,coords[:,2]<zdim)
    obj.filter_system(np.logical_and(np.logical_and(fx,fy),fz))

    shape = shapes(obj, nanoshape, **shape_kwargs)
    shape.cut()

    obj.timeframes[0]['boxsize']= np.array([xdim,ydim,zdim])

    if make_stoich:
        make_stoichiometric(obj,shape,type1,type2,ntype1,ntype2,surf_thick)

    obj.timeframes[0]['boxsize'] = np.array([xdim,ydim,zdim])
    coords = obj.get_coords(0)
    obj.timeframes[0]['coords']+= obj.get_box(0)/2 - mdp.CM(coords,obj.atom_mass)

    if surface_types_map is not None:
        change_surface_types(surface_types_map,obj,shape,surf_thick)

    print('Final system size = {:d}'.format(obj.natoms))
    obj.mol_ids = np.ones(obj.natoms,dtype=int)

    obj.timeframes[0]['coords'] += obj.get_box(0)/2-mdp.CM(obj.get_coords(0),obj.atom_mass)
    return obj,shape



def get_stoichtypes(argstring):
    key1,key2 = argstring.split(',')

    def tn(key):
        ty,nt = key.strip().split(':')
        ty = ty.strip().split('-')
        nt = int(nt)
        return ty,nt
    type1,ntype1 = tn(key1)
    type2,ntype2 = tn(key2)
    return type1,type2,ntype1,ntype2

def get_defaults(command_args):

    type1,type2,ntype1,ntype2 = get_stoichtypes(command_args.types)

    box = np.array(command_args.box)

    stabilizing_springs = command_args.stabilizing_springs

    shape = command_args.shape

    if shape=='tube':
        ka = ['diameter','length','outer_diameter']
        onlyz=False
    elif shape == 'pore':
        onlyz=True
        ka = ['diameter','length']
    elif shape == 'sphere':
        ka = ['diameter']
        onlyz=False
    elif shape=='wrap':
        onlyz=False
        ka = ['diameter','length']
    elif shape =='asis':
        ka =  dict()
        onlyz=True
    kargs = dict()
    for k in ka:
        kargs[k] = getattr(command_args,k)

    return shape,  box, type1, type2, ntype1, ntype2, stabilizing_springs, kargs, onlyz

def particle_Checks(command_args):

    if command_args.shape =='asis':
        return

    if command_args.box is None:
        raise Exception('Box (-b) is required')

    if command_args.shape in ['pore','tube','wrap','sphere']:
        if command_args.diameter is None:
            raise Exception('diameter (-d) is required')

    if command_args.shape in ['pore','tube','wrap']:
        if command_args.length is None:
            raise Exception('length (-l) is required')

    if command_args.shape in ['tube']:
        if command_args.outer_diameter is None:
            raise Exception('outer diameter (-od) is required')

    if command_args.diameter > command_args.box[0]:
        raise Exception('diameter is larger than the box X')
    
    if command_args.diameter > command_args.box[1]:
        raise Exception('diameter is larger than the box y')
    if command_args.shape in ['pore','tube'] and command_args.length  > command_args.box[2]:
        raise Exception('length is larger than than box Z')

    if command_args.shape =='pore' and command_args.diameter > command_args.box[0]-0.5:
        raise Exception('Pore is too thin. Decrease diameter (-d) or increase box (-b) X ')
    
    if command_args.shape =='pore' and command_args.diameter > command_args.box[1]-0.5:
        raise Exception('Pore is too thin. Decrease diameter (-d) or increase box (-b) Y ')

    if command_args.shape=='tube':
        od = command_args.outer_diameter
        if  od is None:
            raise Exception('give outer_diameter(-od) of nanotube')
        elif od > command_args.box[0]:
            raise Exception('outer_diameter(-od) is greater than box X')
        elif od > command_args.box[1]:
            raise Exception('outer_diameter(-od) is greater than box Y')
    return

def main():
    adddef = " [  default: %(default)s ]"
    argparser = argparse.ArgumentParser(description="Make a Nanosystem using bulk solid and bulk polymer material")

    argparser.add_argument('-c',"--configuration_file",metavar=None,
            type=str, required=True,
            help="gro file of equilibrated bulk solid material. This bulk material will be cutted in the specified shape or be wrapped in cylindrical shape. Use a proper surface when wrapping")
    argparser.add_argument('-i',"--itp_files",metavar=None,
            type=str, required=True,
            help="itp file of solid material. Needed to get the bond information and types")

    argparser.add_argument('-top',"--top_file",metavar=None,
            type=str, required=True,
            help="top file of the solid material. Needed it to have the force field")

    argparser.add_argument('-s',"--shape",metavar=None,
            type=str, required=True,
            help="Geometry of cutting the solid material",
            choices=['tube','sphere','pore','wrap','asis'])

    argparser.add_argument('-prefix',"--prefix",metavar=None,
            type=str, required=False, default='',
            help="prefix for saving output directory and files")

    argparser.add_argument(
        "-b", "--box",
        metavar=("X", "Y", "Z"),
        type=float,
        nargs=3,
        required=False,
        help="Box size in nm: X Y Z (e.g. -b 10 10 15)"
    )

    argparser.add_argument('-d',"--diameter",
            type=float, required=False, metavar=None,
            help="diameter of the geometry. For tube is inner")

    argparser.add_argument('-od',"--outer_diameter",
            type=float, required=False, metavar=None,
            help="diameter of the geometry. For tube is inner. For wrapped tube is mean")

    argparser.add_argument('-l',"--length",
            type=float, required=False, metavar=None,
            help="length of pore or nanotube")

    argparser.add_argument('-o',"-outputfile",metavar=None,
            type=str, default = 'shaped.gro',
            help="name of the output file"+adddef)

    argparser.add_argument("-nocut","--nocut", metavar=None,
                           type=bool,default=False,
                           help="Valid when shape is 'wrap' . It does not cut the wrapped surface. Useful to have 'perfect' wrapped surface"+adddef)

    argparser.add_argument("-stoich","--stoichiometry", metavar=None,
                           type=bool,default=True,
                           help="Make the surface stoichiometric"+adddef)

    argparser.add_argument("-sth","--surface_thickness", metavar=None,
                           type=float,default=0.15,
                           help="surface_thickness"+adddef)
    argparser.add_argument("-k","--stabilizing_springs", metavar=None,
                           type=float, required=False, default =-1,
                           help="spring constants for stabilization of the new nanoparticle. Pass -1 for no stabilizing springs")
    argparser.add_argument("-t","--types", metavar=None,
                           type=str, default='Alt-Alo:2,O:3',
                           help="types for stoichiometric calculations "+adddef)



    command_args = argparser.parse_args()
    itps = command_args.itp_files
    itp_files = itps if '[' not in itps or '(' not in itps else eval(itps)
    obj = mdp.Analysis(command_args.configuration_file,itp_files,
                       fftop=command_args.top_file)

    obj.read_file(command_args.configuration_file)

    shape, box, type1, type2, ntype1, ntype2, stabilizing_springs, kargs, onlyz = get_defaults(command_args)

    particle_Checks(command_args)

    if shape!='wrap':
        obj,cs = make_nano(obj, shape,box,
                type1=type1,
                type2=type2,
                ntype1=ntype1,
                ntype2=ntype2,
                surf_thick=command_args.surface_thickness,
                make_stoich=command_args.stoichiometry,
                surface_types_map={'Alo':'Alt'},
                **kargs)
    else:

        obj,cs = wrap_surface(obj, box,
                    nocut=command_args.nocut,
                    type1=type1,
                    type2=type2,
                    ntype1=ntype1,
                    ntype2=ntype2,
                    surface_types_map=None,
                    surf_thick=command_args.surface_thickness,
                    **kargs)
    if stabilizing_springs > 0.0:
        obj.find_locGlob()
        print(f'Adding position restraints k = {stabilizing_springs:5.4f}')
        for mol_name in np.unique(obj.mol_names):
            obj.ff.posres = [{'by':'mol_names','val':mol_name,'k': stabilizing_springs}]
    #Write the files
    name = f'{command_args.prefix}_{command_args.shape}'
    mdp.ass.make_dir(name)
    obj.write_gro_file(f'{name}/{name}.gro')
    obj.write_topfile(f'{name}/{name}.top')

if __name__ =='__main__':
    main()
