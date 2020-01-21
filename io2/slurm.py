#!/usr/bin/env python

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



class Queue(object):

    def __init__(self, obj):

        if isinstance(obj, str):
            li = obj.split()
        elif isinstance(obj, list):
            li = obj
        else:
            raise Exception

        ps = ap.ArgumentParser()

        ps.add_argument('--id', nargs='1', type=str, help='identity of the job series, mandatory as an input')
        ps.add_argument('-n', '-np', '--nslot','--nslots', '--ncpu','--ncpus', '--ncore', '--ncores', dest='ncpu', nargs='?', type=int, default=1, help='number of cores for each job, default:1')
        ps.add_argument('-q', '--queue', dest='q', nargs='1', type=int, help='queue for jobs, must be specified')
        ps.add_argument('-hrs', '--hours', dest='hrs', nargs='1', type=int, help='set hours for the job, mandatory')
        ps.add_argument('-mins', '--minutes', dest='mins', nargs='?', type=int, default=0, help='set minutes for the job, optional')
        ps.add_argument('-nf', dest='nf', nargs='?', type=int, default=1, help='number of files to process for each job, default to 1')
        ps.add_argument('-r', '--reverse', action='store_true', help='reverse order of input files?')
        ps.add_argument('-m', '--mem', nargs='?', type=str, help='memory to be allocated by slurm')
        ps.add_argument('--disk', nargs='?', type=str, help='disk space to be allocated by slurm, e.g., 2G')
        ps.add_argument('-i', '-ipts', '--ipts', dest='ipts', nargs='*', help='input files')

        ag = ps.parse_args(li)
        self.ag = ag

        assert len(ag.ipts)>0, '#ERROR: no input file found! '

        fs = []
        for f in ag.ipts:
            output=f[:-4] + '.out'
            if not os.path.exists(output):
                fs.append(f)
        assert len(fs)>0, '#ERROR: no input file! You may need to delete some out files??'

        self.fs = fs
        if ag.reverse:
            self.fs = fs[::-1]



    def split(self):
        return


    def verify_settings(self):
        """ server dependent """
        hostname = os.environ['HOSTNAME'] #"login10.cluster.bc2.ch" or "alchemy"
        q = self.ag.q
        if 'bc2' in hostname:
            _q = {'long': '1day', 'normal': '6hours', 'short': '30min'}[q]
            #assert self.ag.q in queue.keys(), '#ERROR: queue not one of short,normal,long'
            hrs_max = {'long':'1-00:00:00', 'normal':'6:00:00','short':'0:30:00'}[sq]
            ag.q = _queue[ag.q]
        elif 'alchemy' in hostname:
            queue = sq
            hrs_max = {'long': '48:00:00', 'normal': '2:00:00', 'himem': '24:00:00'}[sq]
        else:
            raise Exception('#ERROR: host %s not supported!!'%hostname)

        shrs = _hrs if ihrs else hrs_max
        hrs = '%s:%s:00'%(shrs, _mins)



        USER = os.environ['USER']
        wd = os.environ['PWD']
        if not os.path.exists(wd+'/trash'): os.mkdir(wd+'/trash')
        #print 'wd = ',wd



qname = identity
fid = open(qname, 'w')
fseed = '%s/%s.cmd'%(wd,qname)

scr = '/scratch/%s/%s'%(USER,qname)

fid.write( "#!/bin/bash\n\n" )
fid.write( '#SBATCH --job-name=%s\n'%qname )
fid.write( "#SBATCH --time=%s\n"%hrs )

if 'bc2' in hostname:
  fid.write( "#SBATCH --qos=%s\n"%queue )
else:
  fid.write( "#SBATCH --partition=%s\n"%queue )

fid.write( "#SBATCH --mem=%s\n"%smem )
fid.write( "#SBATCH --tmp=%s\n"%sdisk )
fid.write( "#SBATCH --array=1-%d\n"%nfs )
fid.write( "#SBATCH --cpus-per-task=%s\n"%ncpu )
fid.write( '#SBATCH --output=%s/trash/%%A_%%a.o\n'%wd )
fid.write( '#SBATCH --error=%s/trash/%%A_%%a.e\n\n'%wd )

fid.write( '# delete files older than 1 days\n' )
fid.write( 'rm -rf $(find /scratch/$USER/ -type d -ctime +1)\n' )
#fid.write( "rm -rf %s\n"%scr )

fid.write( 'SEEDFILE=%s\n'%fseed )
fid.write( 'SEED=$(sed -n ${SLURM_ARRAY_TASK_ID}p $SEEDFILE)\n' )
fid.write( 'eval $SEED\n' )

fid.close()

#if ig09:
#  #fid.write( "module load gaussian\n" )
#  fid.write( "export OMP_NUM_THREADS=1\n" )
#  fid.write( "export GAUSS_SCRDIR=%s\n"%scr )
#fid.write( "if [ -d %s ]\nthen\n  rm -r %s\nfi\n"%(scr,scr) )
#fid.write( "mkdir -p %s\n"%scr)


with open(fseed,'w') as fid2:
  for i in range(nfs):
    inp = fsu[i]
    out = inp[:-4] + '.out'
    if ig09:
      fid2.write( "%s %s %s\n"%(EXE,inp,out) )
    elif imolpro:
      fid2.write( "%s -n 1 -t %s -d%s -I%s -W%s %s\n"%(EXE, ncpu,scr,scr,scr,inp))
    else:
      raise Exception('not implemented yet')

  cmd = 'sbatch %s'%qname; #print cmd
  iok = os.system(cmd)

#  -l membycore=5G -q short.q


