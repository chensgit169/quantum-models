import quantities.constants as qc

for name in dir(qc):
    print(name, getattr(qc, name))
