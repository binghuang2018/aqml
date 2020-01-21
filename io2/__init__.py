
import os,sys,time,re,fnmatch
import numpy as np
import ase.units as au

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(("Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)))
    return

def cmdout(cmd):
    return os.popen(cmd).read().strip().split()

def cmdout1(cmd):
    return os.popen(cmd).read().strip().split('\n')

def cmdout2(cmd):
    return os.popen(cmd).read().strip()

def get_label(k,n):
    fmt = '%%0%dd'%n # equal to  '0'*(nd-len(str(i))) + str(i)
    return fmt%k


class Units(object):
    def __init__(self):
        self.a2b = 1/au.Bohr #1.8897261258369282;
        # bohr to angstrom
        self.b2a = au.Bohr #0.5291772105638411;

        self.h2e = au.Hartree  #27.211386024367243
        self.c2j = au.kcal/au.kJ
        self.j2c = 1.0/self.c2j

        # eV to kcal/mol, kj/mol
        self.e2kc = au.eV/(au.kcal/au.mol) #23.060548012069496;
        self.e2kj = au.eV/(au.kJ/au.mol) #96.48533288249877;

        # kcal/mol, kj/mol to eV
        self.kc2e = 1.0/self.e2kc #0.04336410390059322;
        self.kj2e = 1.0/self.e2kj #0.010364269574711572;

        #
        self.h2kc = self.h2e*self.e2kc #627.5094738898777;
        self.h2kj = self.h2e*self.e2kj #2625.4996387552483;


class Folder(object):

    def __init__(self, wd, format, regexp='', use_conf=True):
        assert os.path.exists(wd), '#ERROR: folder does not exist?'
        self.wd = wd[:-1] if wd[-1] == '/' else wd
        self.format = format

        #nfs = int(os.popen('\ls -afq %s/ | wc -l'%wd).read().strip()) - 2
        # The reason that we have to subtract 2 is due to that
        # "\ls -afq" considers "." and ".." as two files as well

        fs = np.sort( os.listdir(wd) )
        fs2 = []; nss = []
        for f in fs:
            if f.endswith(format):
                if regexp != '':
                    if not fnmatch.fnmatch(f, regexp): continue
                fi = '%s/%s'%(self.wd, f)
                nss.append( len(fi) )
                fs2.append( fi )
        nss_u = np.unique(nss)
        if len(nss_u) > 1:
            # e.g., for water clusters: frag_001.xyz, frag_001_face.xyz (conformer)
            print(' ** WARNING: The lengths of filenames are not the same')
            if not use_conf:
                fs1 = np.array(fs1)
                nss = np.array(nss)
                fs2 = fs1[ nss == np.min(nss_u) ]
        self.fs = fs2
        self.nc = len(fs2) # number of total molecules (including conformers)

    def update_fidx(self):
        """
        sometimes, the idx in the filename is not consistanet with the
        line number shown for `ls -1 frag_*xyz | nl`, e.g.,
           1 frag_001.xyz
           2 frag_003.xyz
           3 frag_004.xyz
        now we want the two numbers to be consistent

        Attention: the renmaing cannot be parallelized!!
        """
        # possible filenames:
        # frag_001.out, frag_001_c0001_raw_u.out, ...
        idxs = np.array([ re.findall('frag_(\d+)[\._]', fi)[0] for fi in self.fs ])
        idxs_u = np.sort(np.unique(idxs))
        self.sidxs = idxs
        self.sidxs_u = idxs_u
        self.nm = idxs_u.shape[0]
        if self.nc > self.nm:
            print(' ** WARNING: there are conformers!!')

        ims = np.arange(self.nm) + 1
        n = len(str(self.nm))
        fsu = []
        for i in range(self.nc):
            f = self.fs[i]
            sidx = self.sidxs[i]
            sidx_u = get_label( ims[ sidx == self.sidxs_u ], n )
            if sidx != sidx_u:
                fu = sidx_u.join( f.split(sidx) )
                fsu.append(fu)
                cmd = 'mv %s %s'%(f,fu)
                iok = os.system(cmd)
            else:
                fsu.append(f)
        self.fs = fsu

    def filter_files(self, idsOnly=[], idsException=[], substitute_fs=[]):
        """
        filter molecules further
        """
        nmu = self.nc
        fsu = self.fs
        vf3 = np.vectorize(retrieve_idx)
        sidxs = vf3(fsu)
        if not use_conf:
            msg = '#ERROR: there are conformers!!'
            assert len(sidxs) == len(set(list(sidxs))), msg
            for fi in substitute_fs:
                sidxi = retrieve_idx(fi)
                iif = sidxs.index(sidxi)
                fsu[iif] = fi

        n = len(idsException)
        if n > 0 and n < nmu:
          if not use_conf:
            nmu = nm1 - n
            idxs1 = np.arange(nm1)
            fsu = fsu[ np.setdiff1d(idxs1, np.array(idsException)-1) ]
          else:
            raise '#ERROR: Plz fill code here to process such case'

        n2 = len(idsOnly)
        if n2 > 0:
          if not use_conf:
            nmu = n2
            idxs2 = np.array(idsOnly) - 1
            fsu = fsu[ idxs2 ]
          else:
            raise '#ERROR: Plz fill code here to process such case'

        #print ' -- use_conf, fsu = ', use_conf, fsu

        return nmu, fsu

def savezWrapper(data, path, compressed = False):
    from numpy import savez,savez_compressed
    if compressed:
        executeSting = "savez_compressed(f"
    else:
        executeSting = "savez(f"
    for key in list(data.keys()):
        executeSting += "," + key + "= data['" + key + "']"
    executeSting += ")"
    with open( path, 'wb' ) as f:
        exec(executeSting)

def loadWrapper(path):
    from numpy import load
    out = {}
    with open( path, 'rb' ) as f:
        data = load(f)
        for key in data.files:
            out[key] = data[key]
    return out

def remove_files(obj):
    """
    delete a file or multiple files
    """
    fs = []
    if type(obj) is str:
        if os.path.exists(obj):
            fs.append(obj)
    elif type(obj) is list:
        for obj_i in obj:
            if os.path.exists(obj_i):
                fs.append(obj_i)
    os.system('rm %s'%( ' '.join(fs) ) )

