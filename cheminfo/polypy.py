#!/usr/bin/python
polypy_version=1.416
import sys, os, re, time, string, argparse, math, itertools
from copy import copy
''' polypy -- python program to calculate rings in a graph, using Franzblau statistics
    (Phys. Rev. B, vol 44, 4925 (1991): D.S. Franzblau, "Computation of ring statistics for network models of solid")

    Copyright (C) 2010-2013 Jaap Kroes (jaapkroes@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    notation used in this program:
    g   -   graph
    r   -   ring, list of vertices that are form ring in g (note, rings also form graphs)
    v   -   vertex
    n   -   neighboring vertex, relative to v
'''
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description='''\
This program uses Franzblau statistics [Phys. Rev. B, vol 44, 4925 (1991)] (with a correction since v1.4) to find irreducible rings in molecular structures. 

Options include
- searching for rings, 
- searching for chains, 
- printing the coordination of each atom, 
- calculating improper dihedrals of atoms with more than 3 neighbors, 
- writing an output marked according to ringsize of coordination, 
- writing simple output for large multi-framed files
''')
#define command line options
parser.add_argument(dest="geo",type=str,help="input xyz file")
parser.add_argument("-c", "--chainbool", action="store_true", help="identify chains of atoms: defined as a connected line of 2n-atoms (default=false)")
parser.add_argument("-d", "--depth", type=int, default=7, help="maximum ring size, computation time is O(x^d), where x is average coordination. (default=7), note: 0=infinite depth")
parser.add_argument("-g", "--cluster_rings", dest="clusterbool", action="store_true", help="group/cluster connected rings (default=false)")
parser.add_argument("-m", "--mark", type=int, default=0, help="determine output marking criterium : 0 (none, default), 1 (neighbors), 2 (ring-size), 3 (chains), 4, 5, 6 (same but in PDB format instead of XYZ)")
parser.add_argument("--polygon", action="store_true", help="write polygon file for 3D visualization")
parser.add_argument("-b", "--pbc", action="store_true", help="use periodic boundary conditions, cell size on 2nd line of xyz-file as Lx Ly Lz")
parser.add_argument("-l", "--ringbool", action="store_false", help="switch on/off: search for rings/loops (default=true)")
parser.add_argument("-p", "--impropbool", action="store_true", help="calculate improper angles (default=false)")
parser.add_argument("--improplimit", type=float, default=0, help="lower limit for counting improper angles in simple NFO (default=0)")
parser.add_argument("--n2r6", action="store_true", help="calculate number of \"SP\"-atoms (two neighbors in a hexagon) (default=false)")
parser.add_argument("-f", "--floodfill", action="store_true", help="find disconnected subgraphs (default=false)")
parser.add_argument("-o", "--origin", type=int, default=-1, help="find neighbor-shells of atom i (disabled if < 0)")
parser.add_argument("--connect", type=str, help="define explicit connections, file-format: one line per atom, defining all neighbors for this atom (indices run from 0 to n-1)")
parser.add_argument("-i", "--info", action="store_false", help="write nfo file (default=true)")
parser.add_argument("--stride", default=1, type=int, help="stride xyz file")
parser.add_argument("-s", "--simple", action="store_true",help="print simple nfo file, prints only ring counts (default=false)")
parser.add_argument("-q", "--quiet", action="store_true",help="suppress progress bars (default=false)")
parser.add_argument("-x", "--franzblau", action="store_false", help="removed rings that are not shortest path rings (default=true)")
parser.add_argument('--verbose', '-v', action='count')
parser.add_argument('--version', action='version', version="%(prog)s version " + str(polypy_version))
######### MAIN #########
def main():
  args = parser.parse_args()
  starttime = time.time()
  basename = args.geo[:-4]
  d = args.depth
  if args.verbose: sys.stderr.write("Starting polypy %s: file = %s, depth = %d\n" %(polypy_version, args.geo, d))
  data,max_radius = loadxyz(args.geo) # load data
  if args.verbose: sys.stderr.write("Data loaded from %s : max atomic radius = %f\n"%(args.geo,max_radius))
  nframes = len(data) # number of frames

  if args.info or args.simple:
    nfo_file = open(basename+'.nfo','w')
    nfo_file.write('################################## \n')
    nfo_file.write('# polypy %s\n'%polypy_version)
    nfo_file.write('# file:         '+args.geo+'\n')
    nfo_file.write('# current path: '+os.path.abspath('./')+'\n')
    nfo_file.write('# frames:       '+str(nframes)+'\n')
    nfo_file.write('# max_depth:    '+str(d)+'\n')
    nfo_file.write('# max_radius:   '+str(max_radius)+'\n')
    nfo_file.write('# arguments:    '+str(args)+'\n')
    nfo_file.write('################################## \n')
    if args.simple: # write header for simple nfo-file : columns for coordination, rings, chains
      nfo_file.write("#frame 0n   1n   2n   3n   4n")
      dstrings = d if d>2 else 10
      for m in range(3,dstrings+1): nfo_file.write(" %3ir" % int(m))
      if args.n2r6: nfo_file.write("   2n6r")
      if args.impropbool: nfo_file.write("  #IMPR")
      if args.chainbool: nfo_file.write("  nc  lc")
      nfo_file.write("\n")
  
  # evaluate data
  if args.mark>0 and args.mark<=3: marked_file=open(basename+'_marked.xyz','w') # remove old data
  for i,snap in enumerate(data):
    if not i%args.stride==0: continue
    if(nframes>1): sys.stderr.write("FRAME %d / %d\n"%(i,nframes))
    if args.quiet: g = graph(snap)                    # initialize a graph based on data
    else: g = graph(snap, progressbar(25))
    if args.pbc: g.box = snap['cell']                 # set cell size
    if args.connect: g.connect_file(args.connect)     # read connections from file
    else: g.connect_sparse(max_radius,args.pbc)       # connect graph (based on atomic radii)

    if args.origin > -1: g.shells(args.origin,d)      # print neighbor shells
    if args.ringbool: 
      g.polycount(d)                                  # find rings
      if args.franzblau: g.remove_filled_polygons()   # remove non-elementary rings
    if args.chainbool: g.find_chains()                # find chains 
    if args.clusterbool: g.find_clusters()            # cluster rings
    if args.floodfill: g.floodfill()                  # find connected subgraphs
    if args.impropbool: g.improp(args.pbc)

    if args.mark:
      if args.mark==1 or args.mark==4: g.mark_neighbors(args.mark)    # mark atoms of different coordination by atom type
      if args.mark==2 or args.mark==5: g.mark_rings(args.mark)        # mark elements of ring by different atom type
      if args.mark==3 or args.mark==6: g.mark_chains(args.mark)       # mark atoms in chains as Helium (He)
      if args.mark>3:   g.writepdb(open(basename+'_marked.pdb','w'),args.pbc,"-- marked pdb file --")
      elif args.mark>0: g.writexyz(marked_file,"-- marked xyz file --")
      if args.polygon: g.writepolygons(open('polygons.dat','w'))

    if args.simple:
      nfo_file.write("%-6d" % i+' ')
      g.print_simple_info(nfo_file,d,args)            # print info to file
    elif args.info: g.print_info(nfo_file,args)       # print extended info to file

  if args.verbose: sys.stderr.write("Finished in %.2f seconds.\n"%(time.time()-starttime))
  return 0
####### END MAIN ########

class vertex:
  ''' Vertices represent the atoms
  A vertex is some object with neighbors.
  It is usually given an ID so we can print this to stdout.
  Position is used to see if atoms are connected.
  '''
  def __init__(v, id = 0, pos = [], atom = 'C'):
    v.neighbors = [] # list of neighbors, e.g. [1,2,6] (actually they are only pointers to other vertices)
    v.neighborscopy = []
    v.nn = 0         # number of neighbors
    v.id = id        # site id, used for convenience of printing
    v.pos = pos      # position, [x,y,z]
    v.atom = atom    # atom label (element of periodic table)
    v.mark = 0       # mark for pdb
    v.ingraph = True # flag if atom was visited, setting to false effectively removes it from the graph
    v.visited = False 

  def __str__(v):
    return "vertex id %d, type %s, position [%.3f, %.3f, %.3f]" % (v.id,v.atom,v.pos[0],v.pos[1],v.pos[2])

class graph:
  ''' A graph is a set of vertices
  Every graph has a size (the number of vertices/atoms) and a list of vertices.
  Based on this information we can calculate rings in the graph.
  '''
  def __init__(g, data = None, progress = None):
    g.vertices = []
    g.size = 0
    if data:
      for i,coords in enumerate(data['coords']):
        v = vertex(i,coords,data['types'][i])
        g.vertices.append(v)
        g.size = data['size']
    g.chains = []
    g.rings = [] 
    g.path = []
    g.improper = []
    g.progress = progress

  def connect_file(g, filename):
    with open(filename,'r') as f:
      for line in f:
        indices = [ int(index) for index in line.split() ]
        i = indices[0]
        if i>=len(g.vertices) : sys.exit("ERROR index out of range : "+str(i))
        v = g.vertices[i]
        for j in indices[1:]:
            if j>=len(g.vertices) : sys.exit("ERROR index out of range : "+str(j))
            n = g.vertices[j]
            if n not in v.neighbors: 
                v.neighbors.append(n)
                v.nn += 1
    for v in g.vertices: # make a backup
      for n in v.neighbors:
        v.neighborscopy.append(n)

  def connect_sparse(g, max_radius, pbc):
    ''' 
    algorithm Verlet lists:     this algorithm is O(n) even for non-packed structures, see below
                                using the sparsity of the space (with dicts, since polypy 1.2), the memory usage is also O(n)
                                since polypy 1.4 the algorithm has improved speed thanks to 'cProfile'
                                usage of python profiler: python -m cProfile polypy testfile.xyz

    1. find a box surrounding all atoms (h1,h2,h3),                     O(1) 
    2. divide the box into small boxes with size  max_r,                O(n)    for n atoms
    3. now for all elements in one of these boxes,                      O(n*d)  for average density d
        3a. find neighbors based on box itself and neighboring boxes,   O(1)
    '''
    if g.progress: g.progress.updateinitial('neighbors','-+')
    # clear old connections
    for v in g.vertices:
        del v.neighbors[:]
    # 3D Verlet lists:
    #  coordinates
    x = [v.pos[0] for v in g.vertices] 
    y = [v.pos[1] for v in g.vertices] 
    z = [v.pos[2] for v in g.vertices] 
    minx=min(x); miny=min(y); minz=min(z) # minimum x,y,z
    maxx=max(x); maxy=max(y); maxz=max(z) # maximum x,y,z
    # box dimensions
    if pbc:
      nx = int(g.box[0]/max_radius); 
      ny = int(g.box[1]/max_radius); 
      nz = int(g.box[2]/max_radius); 
      if nx is 0 or ny is 0 or nz is 0: sys.exit("using PBCs with too small or undefined cell for this frame")
    else: nx=ny=nz=0
    # Now we use that the space can be sparsely filled, so that we do not want to allocate memory for every box
    # e.g. in a graphene structure with one atom far above the flat sample, all boxes (except 1) above the sample are empty..
    vl = dict()
    numtotal = 0
    for v in g.vertices:
      # place element in box with indices n1,n2,n3
      key = (int((v.pos[0]-minx)/max_radius),int((v.pos[1]-miny)/max_radius),int((v.pos[2]-minz)/max_radius))
      ### if atom is exactly on pbc, place it in vertix 0 else it will not be recognized as a neighbor in line 
      ### 255, if pbc: key2=((i+di)%nx,(j+dj)%ny,(k+dk)%nz) because for this atom i+dj == ny 
      ### (robert.meissner@ifam.fraunhofer.de, 2014-11-21, version 1.414)
      if key[0] is nx: key = (0,key[1],key[2])
      if key[1] is ny: key = (key[0],0,key[2])
      if key[2] is nz: key = (key[0],key[1],0)
      ###
      if key in vl:
        vl[key].append(v)
      else: 
        vl[key] = [v]
        numtotal+=1

    # store information in graph
    g.vl = vl
    g.max_radius = max_radius

    # precalculate where possible
    inv_numtotal = 1./numtotal

    # progress counter to prevent unnecessary progress writes
    current_percentage = 0

    for num,(key,vbox) in enumerate(vl.items()):
      (i,j,k) = key
      if g.progress:
        percentage = num*inv_numtotal
        if int(percentage*100)/100. > current_percentage:
          current_percentage = percentage
          g.progress.update(percentage,'neighbors (%d/%d)'%(num+1,numtotal))
      # for all atoms in this box
      for v in vbox:
        # go over all neighboring boxes (i+di,j+dj,k+dk)
        for di in range(-1,2):
          for dj in range(-1,2):
            for dk in range(-1,2):
              if pbc: key2=((i+di)%nx,(j+dj)%ny,(k+dk)%nz)
              else: key2=(i+di,j+dj,k+dk)
              if key2 in vl:
                for n in vl[key2]:
                  if n != v: # neighbor not equal to self
                    dx = v.pos[0] - n.pos[0]
                    dy = v.pos[1] - n.pos[1]
                    dz = v.pos[2] - n.pos[2]
                    if pbc:
                      if abs(dx)>g.box[0]/2: dx = abs(dx)-g.box[0]
                      if abs(dy)>g.box[1]/2: dy = abs(dy)-g.box[1]
                      if abs(dz)>g.box[2]/2: dz = abs(dz)-g.box[2]
                    rij2 = dx*dx + dy*dy +dz*dz
                    if rij2 < r22[v.atom][n.atom]:
                      v.neighbors.append(n)
                      v.nn += 1
    for v in g.vertices: # and make a backup
      for n in v.neighbors:
        v.neighborscopy.append(n)
    if g.progress: g.progress.updatefinal('neighbors (%d/%d)'%(num+1,numtotal))

  ''' find_rings:
  !!! Most important part of the program: here we actually search for the rings !!!
  parameters:
  v           - a vertex that from where search starts 
  visited     - keeps track of where we have been
  depth       - number of steps taken (equals number of items in 'visited')
  root        - root node from where search started, not equal to self because of recursivity
  rings       - saves the complete set of rings found thus far
  ''' 

  def find_rings(g, v, rings = [], visited = [], depth = 0, root = None, max_depth = None):
    ## initialization
    if root is None:          # At the first call:
      root = v                # - set root to current node and
      root.ingraph = False    # - remove root from the graph
    if max_depth: # max_depth to search (None means search until any length, max = #atoms)
      if depth >= max_depth: return False
    visited.append(v) # add this point to visited points
  
    ''' algorithm depth-first search:
    1. pick root node
    2. search all neighboring nodes
    3. start recursion (from 2nd neighbors)
      3a. if neighbor equals root: ring found!
      3b. else search all (unvisited) neighbors
    4. remove root node from the graph
    5. define a list with rings in graph (g.rings)
    '''
    ## start searching
    depth += 1 # go one layer below
    for n in v.neighbors:           # for all neighbors
      if depth > 2 and n is root:   # did we find a ring?
        rings.append(copy(visited)) #  -> if so add a copy to the set of rings
      # otherwise we search all neighbors that haven't been searched already
      elif n.ingraph:
        n.ingraph = False
        g.find_rings(n, rings, visited, depth, root, max_depth)
        n.ingraph = True
    # !! BUG FIXED (v1.4), this error is also in [PRB 44-4925 (1991)]
    # don't remove edge, remove only root from neighbors of v
    if depth == 2:
      if root in v.neighbors:
        v.neighbors.remove(root)
    visited.pop()

  def shells(g, origin, max_depth):
      o = g.vertices[origin]
      sys.stdout.write("# nth neighbors of (%s)\n"%o)
      oneighbors = g.search_from_origin(o,max_depth)
      for d in oneighbors:
        sys.stdout.write("[ n = %d ] " % d)
        for v in sorted(oneighbors[d], key = lambda x : x.id):
          sys.stdout.write("%d "%(v.id+1))
        sys.stdout.write("\n")

  def polycount(g, max_depth):    # find the rings from every atom 
    if g.progress: g.progress.updateinitial('rings','o')
    for i in range(g.size):
      if g.progress and i%int(g.size/100.+1)==0: g.progress.update((float(i+1)/float(g.size)),'rings %d/%d'%(i+1,g.size))
      g.find_rings(g.vertices[i], rings = g.rings, max_depth = max_depth)
    if g.progress: g.progress.updatefinal('rings (%d/%d)'%(g.size,g.size))

    for v in g.vertices:          # restore backup of edges
      v.neighbors = copy(v.neighborscopy)

  def shortest_path(g, v, goal, max_depth, visited = [], depth = 1):
    if depth < max_depth:         # start searching
      depth += 1                  # go one layer below
      visited.append(v)           # add this point to visited points
      if v == goal: 
        lp = len(g.path)
        if depth < lp or not lp:  # current path shorter or first path found
          g.path = copy(visited)
          max_depth = depth
      else:
        for n in v.neighborscopy: # search all neighbors 
          if n.ingraph:           # not already searched
            n.ingraph = False
            g.shortest_path(n, goal, max_depth, visited, depth)
            n.ingraph = True
      visited.pop()

  def remove_filled_polygons(g):  # reduce rings by removing non-SP rings
    if g.progress: g.progress.updateinitial('franzblau','#')
    for v in g.vertices: 
      v.ingraph = True            # make sure all are in the graph before SP-search
    size = len(g.rings)
    rings = g.rings
    to_be_removed = []
    for i in range(size):         # for every ring, r
      if g.progress and i%int(size/100.+1)==0: g.progress.update((float(i)/size),'franzblau (%d/%d)'%(i,size))
      r = g.rings[i]; l=len(r)
      remove = False
      for j in range(l):          # for vertex j in r
        for k in range(j+2,l):    # connect with all other, skip j-j (distance=0) and j-(j+1) (nearest neighbors)
          if not remove:
            v = r[j]; n = r[k]
            djk = abs(j-k)
            dist_r = min(djk,abs(djk-l))+1  # distance over the ring
            g.path = []
            g.shortest_path(v, n, depth = 0, max_depth = dist_r) # SP search between v and n
            dist_g = len(g.path)  # distance over the entire graph
            if dist_g < dist_r:
              remove = True
      if remove: to_be_removed.append(r)
    for r in to_be_removed:
      g.rings.remove(r)
    if g.progress: g.progress.updatefinal('franzblau (%d/%d)'%(size,size))

  def find_chains(g):
    # make subset of 2n vertices
    if g.progress: g.progress.updateinitial("chains",'c')
    begin = []
    g2 = graph()
    for v in g.vertices:
      if v.nn == 2:
        begin.append([v])
    n = len(begin)
    g.chains = []
    if n==0: return
    sys.setrecursionlimit(2*n) # maximum n merges and n copies to end
    end = []
    g.group_chains(begin,end)
    for c in end:
      if c:
        # minimum chain size is 3
        if len(c) > 2:
         g.chains.append(c)
  
  def group_chains(g,begin,end,iter=0):
    iter += 1
    n = len(begin)
    mergebool1 = False
    for i,c1 in enumerate(begin):
      if c1:
        if g.progress: g.progress.update(float(i)/float(n),'chains iter %d (%d/%d)'%(iter,i,n))
        mergebool2 = False
        for j,c2 in enumerate(begin):
          if c2 and i != j:
            if g.chain_overlap(c1,c2):
              # if two chains have overlap: merge
              begin[i] += c2
              begin[j] = None
              mergebool1 = True
              mergebool2 = True
        if not mergebool2:
          end.append(c1)
          begin[i] = None
    if g.progress: g.progress.updatefinal('chains iter %d (%d/%d)'%(iter,n,n))
    if mergebool1:
      g.group_chains(begin,end,iter)  # restart grouping
    else: return
  
  def chain_overlap(g,c1,c2):
    for v1 in c1:
      for v2 in c2:
        if v1 in v2.neighbors: return True
    return False

  def find_clusters(g):
    '''
    A defect is described as a cluster of atypical rings 
    (i.e. in graphitic material, all deviations from hexagons). 
    Two rings are clustered if they have at least one vertex in common. 
    In this algorithm, grouping is done recursively.
    '''
    n = g.size
    g.clusters = []
    if n==0: return
    sys.setrecursionlimit(2*n) # maximum n merges and n copies to end
    clusters = []
    for r in g.rings:
      if len(r)!=6:
        i = len(clusters)
        size = len(r)
        c = dict(id = i, type = (str(size)+'r'), size = size, elements = r)
        clusters.append(c)
    g.group_clusters(clusters,g.clusters) # group rings into connected structures
  
  def group_clusters(g,begin,end):
    # if begin is empty we are done
    if len(begin)==0:
      return 
    r1 = begin[0] # first ring in begin
    for i,r2 in enumerate(begin[1:]):
      if g.overlap_rings(r1,r2):
        # if two rings have overlap
        begin.append(g.merge_clusters(r1,r2))  # merge
        begin.pop(i+1)              # remove old r2
        begin.pop(0)                # remove old r1
        g.group_clusters(begin,end) # restart grouping
        return            # return if recursion jumps back
    # if r1 has no overlap with any other ring
    end.append(r1)    # r1 is final
    begin.pop(0)      # remove r1 from begin
    g.group_clusters(begin,end) # restart grouping
    return            # return if recursion jumps back
    
  def overlap_rings(g,r1,r2):
    for e1 in r1['elements']:
      if e1 in r2['elements']:
        return True
    return False
  
  def merge_clusters(g,r1,r2):
    type = r1['type'] + r2['type']
    elements = []
    size = r1['size']
    for e1 in r1['elements']:
      elements.append(e1)
    for e2 in r2['elements']:
      if e2 not in elements:
        elements.append(e2)
        size += 1 
    return dict(type = type, size = size, elements = elements)

  def floodfill(g):
      if g.progress: g.progress.updateinitial("floodfill",'f')
      # this is an iterative (queue-based) version of
      # floodfill to avoid stack-overflow
      nremoved = 0
      for v in g.vertices: v.ingraph = True
      start = copy(g.vertices)
      subg = 1
      while start:
        q = [start[0]]
        while q:
          v = q.pop()
          v.ingraph = False
          v.subgraph = subg
          if g.progress:
            nremoved += 1
            if nremoved % int(g.size/.100+1)==0: 
              g.progress.update(nremoved/g.size,"floodfill (%d/%d)"%(nremoved,g.size))
          for n in v.neighbors:
            if n.ingraph: 
              n.ingraph = False
              q.append(n)
        subg += 1
        # this last line makes it O(n^2)..
        start = [v for v in g.vertices if v.ingraph]
      if g.progress: g.progress.updatefinal("floodfill (%d/%d)"%(nremoved,g.size))
 
  def improp(g,pbc):
    if g.progress: g.progress.updateinitial("improper",'I')
    # calculate the improper dihedral angle (ijkl)
    # with the vertex as the central atom (i)
    SMALL = 0.001
    nimprop = 0
    for v in g.vertices:
      if v.nn > 2:  
        # run over all planes for central vertex v spanned by 3 neighbors
        theta = []
        i = v
        for plane in itertools.combinations(range(v.nn),3):
          pneighs = v.neighbors[plane[0]],v.neighbors[plane[1]],v.neighbors[plane[2]]
          ptheta = []
          for n in range(3):
            # left shift of neighbor tuple
            nid = list(itertools.islice(itertools.cycle(plane), n, len(plane)+n))
            j = v.neighbors[nid[0]]
            k = v.neighbors[nid[1]]
            l = v.neighbors[nid[2]]

            rij = [a_i - b_i for a_i, b_i in zip(i.pos, j.pos)]
            rkj = [a_i - b_i for a_i, b_i in zip(k.pos, j.pos)]
            rlk = [a_i - b_i for a_i, b_i in zip(l.pos, k.pos)]

            if pbc:
              rij = minimum_image(rij,g.box)
              rkj = minimum_image(rkj,g.box)
              rlk = minimum_image(rlk,g.box)
            
            Rij = math.sqrt(sum([p*q for p,q in zip (rij,rij)]))
            Rkj = math.sqrt(sum([p*q for p,q in zip (rkj,rkj)]))
            Rlk = math.sqrt(sum([p*q for p,q in zip (rlk,rlk)]))

            c0 = (rij[0] * rlk[0] + rij[1] * rlk[1] + rij[2] * rlk[2]) / Rij / Rlk
            c1 = (rij[0] * rkj[0] + rij[1] * rkj[1] + rij[2] * rkj[2]) / Rij / Rkj
            c2 = -(rlk[0] * rkj[0] + rlk[1] * rkj[1] + rlk[2] * rkj[2]) / Rlk / Rkj

            s1 = 1.0 - c1*c1
            if s1 < SMALL: s1 = SMALL
            s1 = 1.0 / s1

            s2 = 1.0 - c2*c2
            if s2 < SMALL: s2 = SMALL
            s2 = 1.0 / s2

            s12 = math.sqrt(s1*s2)
            c = (c1*c2 + c0) * s12

            if c > 1.0: c = 1.0
            if c < -1.0: c = -1.0
            ptheta.append(180.0*math.acos(c)/math.pi)
          theta.append([pneighs,ptheta])
        # store central vertex, neighbor vertices and 
        # theta of planes spanned by all neighboring vertices
        g.improper.append([i,theta])
      if g.progress:
        nimprop += 1
        if nimprop % int(g.size/.100+1)==0: 
          g.progress.update(nimprop/g.size,"improper (%d/%d)"%(nimprop,g.size))
    if g.progress: g.progress.updatefinal("improper (%d/%d)"%(nimprop,g.size))
 
  def search_from_origin(g,v,max_depth):
    # breadth-first-search (BFS) 
    Q = []
    neighbors = {}
    for vertex in g.vertices: vertex.ingraph = True
    v.ingraph = False
    for vertex in v.neighbors: 
      vertex.ingraph = False
      Q.append({'v':vertex,'d':1,'i':vertex.id})
    while Q != []:
      t = Q.pop(0)
      d = t['d']
      if d > max_depth: 
        return neighbors
      if not d in neighbors: neighbors[d] = []
      neighbors[d].append(t['v'])
      for n in t['v'].neighbors:
        if n.ingraph:
          n.ingraph = False
          Q.append({'v':n, 'd':t['d']+1,'i':n.id})
    return neighbors

  def mark_rings(g,mark):
    # initialize markings 
    if mark<=3:
        for v in g.vertices: v.atom = elements[0]
    # ring list (set->list), sort inverted by ring size 7,7,7,...,6,6,6...,5,5,...
    rl = sorted(g.rings, key = lambda r: len(r))
    # this has some ambiguity because one atom can be part of multiple rings
    # rings are marked according to the order in 'rl', which is based on size (...,8,7,6,5,4,3)
    for r in rl:
      if len(r)==6: continue
      for v in r:
        # label size by periodic table element
        if mark>3: v.mark = len(r)
        else: 
            if len(r) < len(elements): v.atom = elements[len(r)]
            else: v.atom = elements[0]

  def mark_neighbors(g,mark):
    # mark atoms by number of neighbors:
    for v in g.vertices:
      if mark>3: v.mark = v.nn
      else: v.atom = elements[v.nn+3] # +3 so that graphitic atoms (3nn) are marked as carbon
 
  def mark_chains(g,mark):
    for v in g.vertices:
      if v.nn == 2:
        if mark>3: v.mark=1
        else: v.atom = "He" # so that chain atoms (2nn) are marked as He, not to interfere with ring-marking
      else:
        if mark>3: v.mark=0
    
  def writexyz(g, file, header):
    file.write("%5i\n" % g.size) # header1 (# atoms)
    file.write(header+"\n")
    for v in g.vertices: file.write("%2s %12.8f %12.8f %12.8f\n" % (v.atom, v.pos[0], v.pos[1], v.pos[2]))

  def writepdb(g, file, pbc, header):
    if pbc: file.write("REMARK\nCRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n"%(g.box[0],g.box[1],g.box[2]))
    else : file.write("REMARK\nCRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1\n"%(100,100,100))
    for i,v in enumerate(g.vertices):
        file.write("ATOM%7d %2s   XXX     1    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(i,v.atom,v.pos[0],v.pos[1],v.pos[2],v.nn,v.mark))

  def writepolygons(g, file):
    for r in g.rings:
      l = len(r)
      file.write('color %d\n'%l)
      c = [0,0,0]
      for v in r:
        for i in range(3): c[i] += v.pos[i]/float(l)
      for i,v in enumerate(r):
        n = r[(i+1)%l]
        x = v.pos
        y = n.pos
        file.write('graphics top triangle {%.3f %.3f %.3f} {%.3f %.3f %.3f} {%.3f %.3f %.3f}\n'%\
              (x[0],x[1],x[2],y[0],y[1],y[2],c[0],c[1],c[2]))

  def print_info(g, f, args):
    f.write('=========== Statistics:\n')
    
    # element neighbor list
    enl = [v.nn for v in g.vertices]
    nl = [sorted([n.id for n in v.neighbors]) for v in g.vertices]

    #### SUMMARIZED DATA ####
    if args.ringbool:
      # ring list: 
      # - ring element sorting by id [5,7,1] -> [1,5,7] 
      # - ring list sorting by size
      # - for rings of same size sort by 
      # - sort ring inter all unique rings, sorted by size and id
      rl = [sorted([int(v.id) for v in r]) for r in g.rings]
      # len(x) is the main key, x[0] is the subkey, etc. (note a ring always has at least 3 elements
      rl = sorted(rl, key=lambda x: (len(x),x[0],x[1],x[2])) 
      
      # ring size list 
      rsl = [len(l) for l in rl] 
      
      # all loops and their size
      for i in range(max(rsl+[0])+1): # i=0,1,2,...,max(rsl)
        c = rsl.count(i) # number of times a ring of size i is found
        if c != 0: f.write(str(c)+' loops of size '+str(i)+'\n')
      f.write('\n')
    if args.chainbool:
      chains = sorted(g.chains, key=lambda x: (len(x),x[0].id)) 
      csl = [len(c) for c in chains] 
      # all chains and their size
      for i in range(max(csl+[0])+1): # i=0,1,2,...,max(csl)
        c = csl.count(i) # number of times a ring of size i is found
        if c != 0: f.write(str(c)+' chains of size '+str(i)+'\n')
      f.write('\n')
    if args.clusterbool:
      typelist = [c['type'] for c in g.clusters]
      clusdict = {} 
      for t in typelist: 
        if not t in clusdict:
          clusdict.update( { t : typelist.count(t) } )
          f.write(str(clusdict[t])+" clusters of type %s\n" % t)
      f.write('\n')

    if args.floodfill:
      subgraphdict = {}
      for v in g.vertices:
        if v.subgraph in subgraphdict:
          subgraphdict[v.subgraph].append(v.id)
        else: 
          subgraphdict[v.subgraph] = [v.id]
      f.write(str(len(subgraphdict))+' subgraphs \n')
      for key,item in subgraphdict.items():
        f.write(str(key)+': ')
        for i in item:
          f.write(str(i)+' ')
        f.write('\n')
      f.write('\n')

    # number of elements with n neighbors
    for i in range(max(enl+[0])+1): # i=0,1,2,...,max(enl) 
      c = enl.count(i) # number of times a ring of size i is found
      if i == 1:
        f.write(str(c)+' elements with '+str(i)+' neighbor\n')
      else:
        f.write(str(c)+' elements with '+str(i)+' neighbors\n')
    if args.n2r6: 
        # get elements with only 2 neighbors which are in a ring 
        # (they are not sp, they are sp2) 
        n2r6 = []
        for v in g.vertices:
          if v.nn == 2:
            #check if atom is in 6r
            for l in rl:
              if v.id in l and len(l) == 6:
                n2r6.append(v.id) 
        # print number of atoms with two neighbors and in six fold ring
        f.write('%s elements with 2 neighbors are in loop of size 6\n' % len(n2r6))


    #### EXTENSIVE DATA ####
    f.write('=========== Extensive data:\n')
    if args.ringbool:
      # list of rings (rl) with all unique rings, sorted by size and id
      f.write('(size) [ring elements]:\n')
      for l in rl:
        f.write('('+str(len(l))+") [")
        for li in l:
          f.write(str(li)+' ')
        f.write("]\n")
      f.write("\n")
    
    if args.chainbool:
      f.write('/size/ [chain elements]:\n')
      for c in chains:
        f.write('/'+str(len(c))+'/ ')
        for ci in c:
          f.write(str(ci.id)+' ')
        f.write('\n')
      f.write('\n')

    if args.clusterbool:
      f.write('<size> type [cluster elements]:\n')
      for i,c in enumerate(g.clusters):
        f.write("<%d> %s [" % (i,c['type']))
        for j in c['elements']:
          f.write("%d "%j.id)
        f.write("]\n")
      f.write('\n')
      
    if args.impropbool:
      f.write('{central element (j)} [plane elements] (theta_ijkl,theta_kjli,theta_ljik) <theta>:\n')
      for v,pl in g.improper:
        for npl,theta in pl:
          f.write("{%s} ["%v.id)
          for np in npl:
            f.write("%s "%np.id)
          f.write("] (")
          mtheta = 0.0
          for ptheta in theta:
            mtheta += ptheta
            f.write("%.2f "%ptheta)
          mtheta /= len(theta)
          f.write(") %.2f\n"%mtheta)
      f.write("\n")

    # list of elements with number of neighbors
    f.write('{element} #neighbors [neighbor list]\n')
    for i in range(g.size):
      f.write('{'+str(i)+'} '+str(enl[i])+' [')
      for nli in nl[i]:
        f.write(str(nli)+' ')
      f.write("]\n")
    f.write("\n")

  def print_simple_info(g, f, max_depth, args):
    # ring list: 
    # - ring element sorting by id [5,7,1] -> [1,5,7] 
    # - ring list sorting by size
    # - for rings of same size sort by 
    # - sort ring inter all unique rings, sorted by size and id
    rl = [sorted([int(v.id) for v in r]) for r in g.rings]
    # len(x) is the main key, x[0] is the subkey, etc. (note a ring always has at least 3 elements
    rl = sorted(rl, key=lambda x: (len(x),x[0],x[1],x[2])) 
    
    # ring size list 
    rsl = [len(l) for l in rl] 
    # element neighbor list
    enl = [v.nn for v in g.vertices]
    
    for i in range(0,5): f.write("%-4d" % enl.count(i)+' ') # 0n,1n,2n,3n,4n
    if not max_depth : max_depth = max(rsl)
    for i in range(3,max_depth+1): f.write("%-4d" % rsl.count(i)+' ') # 3r,4r,5r,...,mr
    if args.n2r6: 
        # get elements with only 2 neighbors which are in a ring 
        # (they are not sp, they are sp2) 
        n2r6 = []
        for v in g.vertices:
          if v.nn == 2:
            #check if atom is in 6r
            for l in rl:
              if v.id in l and len(l) == 6:
                n2r6.append(v.id) 
        # print number of atoms with two neighbors and in six fold ring
        f.write("%-4d" % len(n2r6)+' ')
    if args.impropbool:
        # counter number of improper angles exceeding argument
        nimproper = 0
        for v,pl in g.improper:
            for npl,theta in pl:
                if sum(theta)/len(theta) > args.improplimit: nimproper+=1
        f.write("%-4d" % nimproper)


    if args.chainbool:
      count = 0
      length = 0
      for c in g.chains:
        count += 1
        length += len(c)
      if count != 0:
        length = float(length)/float(count)
      f.write("%-3d" % count+' ')
      f.write("%-3d" % length+' ')
    f.write('\n')

def loadxyz(filename):
    data = []
    file = open(filename,'r')
    alltypes = set()
    while True:
      # header1 (number of atoms)
      try: size = int(file.readline().strip())
      except: break
      snap = {'size':size, 'cell':[], 'coords':[], 'types':[]}
      # header2 (cell size)
      items = file.readline().split()
      try: snap['cell'] = [float(items[0]),float(items[1]),float(items[2])]
      except: snap['cell'] = [0,0,0]
      # content (atom coordinates)
      for i in range(size):
        items = file.readline().split()
        pos = [float(items[1]), float(items[2]), float(items[3])]
        snap['coords'].append(pos)
        snap['types'].append(items[0])
        alltypes.add(items[0])
      data.append(snap)
    file.close()
    max_radius = max(radius[t1] for t1 in alltypes)
    return data,max_radius

def minimum_image(vec,box):
   if abs(vec[0])>box[0]/2:
     if vec[0] < 0.0: vec[0] += box[0];
     else: vec[0] -= box[0]
   if abs(vec[1])>box[1]/2:
     if vec[1] < 0.0: vec[1] += box[1];
     else: vec[1] -= box[1]
   if abs(vec[2])>box[2]/2:
     if vec[2] < 0.0: vec[2] += box[2];
     else: vec[2] -= box[2]
   return vec
   

class progressbar:
  def __init__(self,minlen=10,symbol='-'):
    import fcntl, termios, struct, os
    self.fd = os.open(os.ctermid(), os.O_RDONLY)
    (self.height,self.width) = struct.unpack('hh', fcntl.ioctl(self.fd, termios.TIOCGWINSZ, '1234'))
    self.symbol = symbol
    self.minlen = minlen

  def update(self,percentage,message):
    import fcntl, termios, struct
    (self.height,self.width) = struct.unpack('hh', fcntl.ioctl(self.fd, termios.TIOCGWINSZ, '1234'))
    message = message.ljust(self.minlen)                    # fill message to minlen with spaces
    meslen = len(message)
    barsize = max(0,int((self.width-meslen)*percentage)-2)  # progress bar size : -2 for [,]
    filsize = self.width - meslen - barsize - 2             # fill with spaces to end of line
    sys.stderr.write('\r'+message+'[')
    for i in range(barsize): sys.stderr.write(self.symbol[i%len(self.symbol)])
    for i in range(filsize): sys.stderr.write(' ')
    sys.stderr.write(']')
    sys.stderr.flush()

  def updateinitial(self,message,symbol=None):
    if symbol: self.symbol = symbol
    self.update(0,message)

  def updatefinal(self,message):
    self.update(1,message)
    sys.stderr.write('\n')

# element names
elements = ["X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po", "At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg"]
# atomic radii
radii = [1.5, 1.2, 1.4, 1.82, 2.0, 2.0, 1.7, 1.55, 1.52,1.47, 1.54, 1.36, 1.18, 2.0, 2.1, 1.8, 1.8, 2.27, 1.88, 1.76,1.37, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.4, 1.39, 1.07,2.0, 1.85, 1.9, 1.85, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06, 1.98, 2.16,2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.72, 1.66,1.55, 1.96, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,1.86, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
radius = dict(zip(elements,radii))
atomnumber = dict(zip(elements,range(len(elements))))

r22 = {}
for e1 in elements:
  r22[e1] = {}
  for e2 in elements: r22[e1][e2]=radius[e1]*radius[e2]

if __name__ == "__main__":
    sys.exit(main())
