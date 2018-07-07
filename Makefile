CFLAGS = `pkg-config --cflags opencv` 
CLIBS = `pkg-config --libs opencv` 

CC= g++ -std=c++11
%: %.cpp
	$(CC) $(CFLAGS) -o $@ $< $(CLIBS)
