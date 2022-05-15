You will find the main code in the file entitled "net.py"
It needs the "data.py" files that converts the picctures from the emnist data base into 
a vector of (256,1)
To make it work you will also need to install the emnist database.
You can download it via the keras library.


You can have a look at the pictures of graphs that are taking stocks of the convergence of
the algorithm in comparison with different values of the parameter.

The picture "contour avec centre" shows how I tried to analyse a script by finding the edges 
with a sobel filter then find the center of the edge to then remove the letter in the letter
like for the letter O and then send it to my neural network to find the coresponding value.
Finally the algorithm sends the lettre read by order of apperance.

you will find a power point that explains everything in a matter of days 