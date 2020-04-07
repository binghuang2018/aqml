#!/usr/bin/env python

import argparse as ap
from math import floor
import argparse as ap

T,F = True,False


cmdout = lambda cmd: os.popen(cmd).read().strip()


def get_job_type(f):
    if len(cmdout('grep -E "^\s?(%%nproc|#)" %s'%f)) > 0:
        job = 'g09'
        EXE = os.environ['GAUSS_EXE']
    elif len(cmdout('grep -E "basis\s*=|geometry" %s'%f)) > 0:
        job = 'molpro'
        EXE = os.environ['MOLPRO_EXE']
    elif len(cmdout('grep -E "\%pal|\*xyz" %s'%f)) > 0:
        job = 'orca'
        try:
            exe = os.environ['orca']
        except:
            exe = os.environ['orca4']
    else:
        raise Exception('#ERROR: input file not one of {g09,molpro??}')
    return job, exe


def time_as_string(hrs):
    """ input time: hours, could be integer/float """
    d = floor( hrs/24. )
    h = floor( hrs - d*24. )
    m = floor( (hrs - d*24. - h)*60 )
    #s = floor( (hrs - d*24. - h)*3600 - m*60 )
    if hrs <= 24.:
        st = '%02d:%02d:00'%(h, m)
    else:
        st = '%d-%02d:%02d:00'%(d, h, m)
    return st


class Job(object):

    def __init__(self, f):

        self.f = f

    @property
    def jobtype(self):
        return get_job_type(self.f)

    @property
    def memory(self):
        """ get memory specified in input file """
        if self.jobtype == 'molpro':
            smem = cmdout("grep ',[0-9][0-9]*,M' %s"%self.f)
            if smem[0] == '!' or smem == '':
                raise Exception('** memory size not specified in input file?')
            _mem = int(re.search(',([0-9][0-9]*),M', smem).groups()[0])
        return _mem

    @property
    def ncpu(self):
        if self.jobtype == 'g09':
            scpu = cmdout("grep nproc %s"%self.f)
            _ncpu = int( scpu.split('=')[1] )
        elif self.jobtype == 'molpro':
            _ncpu = None
        else:
            raise Exception('Todo')
        return _ncpu

    _path = {'molpro': '$MOLPRO_EXE',
              'g09': '$GAUSS_EXE'}

    @property
    def xpath(self):
        """ path to executable file """
        return self._xpath[self.jobtype]



class Queue(object):

    def __init__(self, ag):

        if isinstance(obj, str):
            li = obj.split()
        elif isinstance(obj, list):
            li = obj
        else:
            raise Exception
        self.ag = ag

        self.q = ag.q # it's actually `partition' for bc2 cluster

        # Note that test results indicate Hyper-threading
        # is not helpful at all for Molpro ccsd(t)/pcvqz calculations
        # Even worse, it may increase the computation time! In such a
        # case, load 2 x N slots, but set --ntasks=N and -nt=1 for Molpro,
        # and meanwhile don't specify "-ht" when submitting slurm job
        self.nslots = ag.np * ag.nt

        self.nt = 1 # default: no hyper-threading
        if ag.ht:
            self.nt = ag.nt

        assert len(ag.fs)>0, '#ERROR: no input file found! '

        fs = []
        n0 = len(ag.fs) # num of total fs
        for f in ag.fs:
            output=f[:-4] + '.out'
            if not os.path.exists(output):
                fs.append(f)
        assert len(fs)>0, '#ERROR: no input file! You may need to delete some out files??'

        if ag.reverse:
            fs = fs[::-1]

        if ag.nf == 1:
            self.fs = [ [fi] for fi in fs ]
            nj = len(fs)  # number of jobs
        else:
            self.fs = []
            nj = floor( n0 * 1.0/nf )
            if n0 - nj * nf > 0:
                nj += 1
            for i in range(nj):
                ib = i * nj
                ie = (i + 1) * nj
                if ie > n0: ie = n0
                self.fs.append( fs[ib:ie] )
        self.nj = nj

    @property
    def hostname(self):
        if not hasattrib(self, '_hostname'):
            self._hostname = os.popen('hostname').read().strip()
        return self._hostname


    @property
    def user(self):
        return os.environ['USER']


    @property
    def ifd_srun(self):
        """ is srun supported? """
        return 'srun' in cmdout('which srun')


    def get_server_settings(self):
        """ Settings are server dependent """

        ag = self.ag

        scr0 = '/scratch/%s/'%self.user
        pfs = '/scicore/pfs/%s'%self.user

        self.resv = None
        if 'bc2' in self.hostname:
            aqos = ['30min','6hours','1day','1week']
            atime = [ 0.5,  6,       24,    24*7]
            #       partition   qos            reservation  scratch
            _dct = {'scicore': [    aqos[:3],       None,   scr0 ],
                        'smi': [['projects'],   'chemie',   scr0 ],
                     'bigmem': [        aqos,       None,   pfs  ],
                    }
            # maximal memory
            maxm = {'scicore': 120, 'smi':2048, 'bigmem':1024}[self.p] # GB
            _qos, resv, scr = _dct[self.p]
            if len(_qos) == 1:
                self.q = _qos[0]
            else:
                assert self.q in _qos
            self.resv = resv
            self.scr = scr
            maxt = dict(zip(aqos,atime)) # Maximal time of running
        elif 'alchemy' in self.hostname:
            _dct = {'long': [ [], None, scr0 ],
                    'himem':[ [], None, scr0 ],
                    'highermem':[ [], None, scr0 ],
                    }
            maxm = {'normal':120, 'long':120, 'himem':500, \
                    'highermem':1000} # GB
            maxt = {'long':48, 'normal':2, 'himem':48, \
                    'highermem':48}
            self.scr = scr0
        elif 'daint' in self.hostname:
            raise Exception('Todo')
        else:
            raise Exception('#ERROR: host %s not supported!!'%hostname)

        if (ag.hrs is None) and (ag.mins is None):
            self.t = maxt
        else:
            self.t = ag.hrs + ag.mins/60. # in hours


    @property
    def wd(self):
        return os.environ['PWD']

    @property
    def tmpd(self):
        return self.wd + '/Trash'

    def mktmp(self):
        """ make temporary foder for writing log """
        fd = self.cwd+'/Trash'
        if not os.path.exists(fd): os.mkdir(fd)


    def submit(self):

        fmt = '%%0%dd'%(len(str(self.nj)))
        for i in range(self.nj):
            fs = self.fs[i]
            ni = len(fs)

            if self.array:
                qname = self.id
                fseed = '%s/%s.cmd'%(self.wd, qname)
            else:
                qname = self.id + '_' + fmt%(i+1)

            fid = open(qname, 'w')
            fid.write( "#!/bin/bash\n\n" )
            fid.write( '#SBATCH --job-name=%s\n'%qname )
            fid.write( "#SBATCH --time=%s\n"%hrs )

            if 'bc2' in self.hostname:
              fid.write( "#SBATCH --qos=%s\n"%self.q )
            else:
              fid.write( "#SBATCH --partition=%s\n"%self.q )

            fid.write( "#SBATCH --mem=%s\n"%self.mem )
            fid.write( "#SBATCH --tmp=%s\n"%self.disk )
            fid.write( "#SBATCH --cpus-per-task=%s\n"%self.ncpu )

            if self.ag.array:
                fid.write( "#SBATCH --array=1-%d\n"%ni )
                fid.write( '#SBATCH --output=%s/%%A_%%a.o\n'%self.tmpd )
                fid.write( '#SBATCH --error=%s/%%A_%%a.e\n\n'%self.tmpd )
            else:
                fid.write( '#SBATCH --output=%s/%s.o\n'%self.tmpd )
                fid.write( '#SBATCH --error=%s/%s.e\n\n'%self.tmpd )

            #fid.write( '# delete files older than 1 days\n' )
            #fid.write( 'rm -rf $(find /scratch/$USER/ -type d -ctime +1)\n' )

            fid.write( "scr=%s"%self.scr )
            fid.write( "[[ -d {scr}/ ]] && rm -rf {scr}/*\n\n".format(scr=self.scr) )

            if self.ag.array:
                fid.write( 'SEEDFILE=%s\n'%fseed )
                fid.write( 'SEED=$(sed -n ${SLURM_ARRAY_TASK_ID}p $SEEDFILE)\n' )
                fid.write( 'eval $SEED\n' )

            if Job(fs[0]).jobtype in ['g09']:
              #fid.write( "module load gaussian\n" )
              fid.write( "export OMP_NUM_THREADS=1\n" )
              fid.write( "export GAUSS_SCRDIR=%s\n"%self.scr )

            if self.ag.array:
                with open(fseed,'w') as fid2:
                  for j in range(ni):
                    inp = fs[j]; out = inp[:-4] + '.out'
                    job = Job(inp)
                    exe = job.xpath
                    if job.jobtype in ['g09']:
                      fid2.write( "%s %s %s\n"%(exe, inp, out) )
                    elif job.jobtype in ['molpro']:
                      if self.np > 1 and self.ifd_srun:
                        fid2.write( "srun --mpi=pmi2 --ntasks={np} --exclusive {exe}.exe -t {nt} -d$scr -I$scr -W$scr {f}\n".format(exe=exe, np=self.np, nt=self.nt, f=inp))
                      else:
                        fid2.write( "{exe} -n {np} -t {nt} -d{scr} -I{scr} -W{scr} {f}\n".format(exe=self.xpath, n=self.np, nt=self.nt, scr=self.scr, inp))
                    else:
                      raise Exception('not implemented yet')
                fid2.close()
            else:
                for j in range(ni):
                    inp = fs[j]; out = inp[:-4] + '.out'
                    job = Job(inp)
                    exe = job.xpath
                    if job.jobtype in ['g09']:
                      fid.write( "%s %s %s\n"%(exe, inp, out) )
                    elif job.jobtype in ['molpro']:
                      if self.np > 1 and self.ifd_srun:
                        fid.write( "srun --mpi=pmi2 --ntasks={np} --exclusive {exe}.exe -t {nt} -d$scr -I$scr -W$scr {f}\n".format(exe=exe, np=self.np, nt=self.nt, f=inp))
                      else:
                        fid.write( "{exe} -n {np} -t {nt} -d{scr} -I{scr} -W{scr} {f}\n".format(exe=self.xpath, n=self.np, nt=self.nt, scr=self.scr, inp))
                    else:
                      raise Exception('not implemented yet')

            fid.close()

            # make temp dir
            self.mktmp()

            cmd = 'sbatch %s'%qname
            if self.ag.debug:
                print(cmd) #cmd
            else:
                iok = os.system(cmd)

            #  -l membycore=5G -q short.q



ps = ap.ArgumentParser()

ps.add_argument('-id', nargs=1, type=str, help='Job id, mandatory as an input')

ps.add_argument('-np', '-ncpu', dest='np', nargs='?', type=int, default=1, help='number of cores for each job, default:1')
ps.add_argument('-nt', '-nthreads', dest='nt', nargs='?', type=int, default=1, help='number of threads per MPI task, default:1')
ps.add_argument('-ht', '-hyper-threading', dest='ht', action='store_true', help='turn hyper-threading on')

ps.add_argument('-array', '-a', dest='array', action='store_true', help='Use array job')

ps.add_argument('-p', '-partition', dest='p', nargs=1, type=int, help='partition: scicore/smi/bigmem or other options for bc2; short/normal/long/himem for alchemy')

ps.add_argument('-q', '-queue', dest='q', nargs=1, type=int, help='queue for jobs, must be specified')

ps.add_argument('-hrs', '-hours', dest='hrs', nargs='1', type=int, help='set hours for the job, mandatory')
ps.add_argument('-mins', '-minutes', dest='mins', nargs='?', type=int, default=0, help='set minutes for the job, optional')

ps.add_argument('-nf', dest='nf', nargs='?', type=int, default=1, help='number of files to process for each job, default to 1')

ps.add_argument('-m', '-mem', nargs=1, type=str, help='memory to be allocated by slurm')
ps.add_argument('-disk', nargs=1, type=str, help='disk space to be allocated by slurm, e.g., 2G')

ps.add_argument('-r', '-reverse', action='store_true', help='reverse order of input files?')

ps.add_argument('-debug', action='store_true', help='check submission files!')

ps.add_argument('fs', dest='fs', nargs='*', help='input files')

ag = ps.parse_args(li)


Q = Queue(ag)
Q.submit()



