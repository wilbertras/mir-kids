import os, sys
import numpy as np
import h5py


def split_hdf5(infname, destination, nsplit, rack=0, ch=0, gr=0):
    hud = h5py.File(infname, 'r')
    srack = 'rack_%02d' %rack
    sch = 'channel_%02d' %ch
    conf = hud[srack][sch]['configuration_group%d' %gr]
    data = hud[srack][sch]['continuous_group%d' %gr]
    
    length = len(data)
    size = np.ceil( length/nsplit )
    print( 'Data length:', length, ' Split size: ', size, ' Nsplit: ', nsplit)
    dest_fnames = []
    for i in range(nsplit):
        start = int( i*size )
        stop = int( (i+1)*size )
        if stop>length: stop = length
        trimdata = data[start:stop]
        
        ostem = os.path.splitext( os.path.basename(infname) )[0]
        ofname = ostem + '_seg%03d.h5' %i
        ofname = os.path.join(os.path.dirname(destination), ofname)
        with h5py.File(ofname, 'w') as f_dest:    
            f_dest.create_dataset(conf.name, data=conf)
            f_dest.create_dataset(data.name, data=trimdata)
            f_dest[conf.name].attrs.update( conf.attrs )
            f_dest[data.name].attrs.update( data.attrs )

            f_dest.attrs.update( hud.attrs )
            f_dest[srack].attrs.update( hud[srack].attrs )
            f_dest[srack][sch].attrs.update( hud[srack][sch].attrs )
        dest_fnames.append(ofname)
    
    return np.array( dest_fnames )


if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)

    if argc!=3:
        print( 'Usage: python %s [input file path (HDF5)] [Number of split]' %argvs[0] )
        quit()
    
    infile = argvs[1] # string
    nsplit = int( argvs[2] ) # string to integer
    
    ofnames = split_hdf5(infile, nsplit, rack=0, ch=0, gr=0)
    
    print( 'Created files:', ofnames )

