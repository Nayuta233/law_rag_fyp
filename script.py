f= open("data/Singaporean_law/Penal Code 1871.txt","r")
lines = f.readlines()
w = open("data/Singaporean_law/Penal Code 1871 modified.txt","w")
for line in lines:
    if "." in line[:6]:
        if line[0] == ' ':
            line = "Section"+line
        else:
            line = "Section "+line
    w.write(line)
f.close()
w.close()