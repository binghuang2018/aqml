
#from ase.utils import str
import os, sys

# Copyright (C) 2010 by CAMd, DTU
# Please see the accompanying LICENSE file for further information.

# This file is taken (almost) verbatim from CMR with D. Landis agreement

FIELD_SEPARATOR = "\\"
PARA_START = "\n\n"
PARA_END = "\\\\@"

names = ['', '', 'Computer_system', 'Type_of_run', 'Method', 'Basis_set',
         'Chemical_formula', 'Person', 'Date', '', '', '', '', 'Title', '']
names_compact = ['', '', 'Computer_system', 'Type_of_run', 'Method',
                 'Basis_set', 'Chemical_formula', 'Person', 'Date', '', '', '',
                 '', 'Title', '']

charge_multiplicity = 15


class GaussianReader:

    def auto_type(self, data):
        """ tries to determine type"""
        try:
            return float(data)
        except ValueError:
            pass

        try:
            ds = data.split(",")
            array = []

            for d in ds:
                array.append(float(d))

            return array
        except ValueError:
            pass

        return data

    def __init__(self, filename, istart=0):
        """filename is optional; if not set, use parse to set the content"""
        self.istart = istart
        if isinstance(filename, str):
            fileobj = open(filename, 'r') #b')
        else:
            fileobj = filename
            fileobj.seek(0)  # Re-wind fileobj

        cmd = "grep 'nuclear repulsion energy' %s | tail -n 1 | awk '{print $4}'"%filename
        #enn = float( os.popen(cmd).read().strip() )
        #self.enn = enn

        ## in case special characters are encountered
        #content = ''
        #for li in fileobj:
        #    content += li.decode(errors='ignore')
        content = fileobj.read()

# handles the case that users used windows after the calculation:
        content = content.replace("\r\n", "\n")
        #print('c=',content)

        self.parse(content)

    def parse(self, content):
        from ase.data import atomic_numbers
        self.data = []
        temp_items = content.split(PARA_START)
        seq_count = 0
        for i in temp_items:
            i = i.replace("\n ", "")
            if i.endswith(PARA_END):
                i = i.replace(PARA_END, "")
                i = i.split(FIELD_SEPARATOR)

                new_dict = {}
                self.data.append(new_dict)

                new_dict['Sequence number'] = seq_count
                seq_count += 1
                for pos in range(len(names)):
                    if names[pos] != "":
                        new_dict[names[pos]] = self.auto_type(i[pos])

                chm = i[charge_multiplicity].split(",")
                new_dict["Charge"] = int(chm[0])
                new_dict["Multiplicity"] = int(chm[1])

# Read atoms
                atoms = []
                positions = []
                position = charge_multiplicity + 1
                try:
                  while position < len(i) and i[position] != "":
                    s = i[position].split(",")
                    if len(s) > 4:
                        #print("s = ", s)
                        #assert self.istart != 0, '#ERROR: in the coord section, there are more than 4 entries!'
                        self.istart = 1
                    atoms.append(atomic_numbers[s[0]])
                    positions.append([float(s[self.istart+1]), float(s[self.istart+2]), float(s[self.istart+3])])
                    position = position + 1
                except:
                    # for atom
                    positions.append( [0.,0.,0.] )

                new_dict["Atomic_numbers"] = atoms
                new_dict["Positions"] = positions
# Read more variables
                position += 1
                while position < len(i) and i[position] != "":
                    s = i[position].split('=')
                    if len(s) == 2:
                        new_dict[s[0]] = self.auto_type(s[1])
                    else:
                        print("Warning: unexpected input ", s)
                    position = position + 1
        #self.data[-1]['enn'] = self.enn

    def __iter__(self):
        """returns an iterator that iterates over all keywords"""
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, pos):
        return self.data[pos]
