// temporary demo for showing how to set subdomain ids from a .geo file
// modified from https://firedrakeproject.org/demos/L_domain.geo

x0 = 0;
xBf = 2.;
z0 = 0;
zBf = 20.;
xWf = -20.;
zWf = 10.;
lc = 1.0;

// following indices 1 .. 26 are seen as gmsh "Elementary entity tag"s (in gmsh view)
Point(1) = {x0, z0, 0, lc};
Point(2) = {xBf, z0, 0, lc};
Point(3) = {xBf, zWf, 0, lc};
Point(4) = {xBf, zBf, 0, lc};
Point(5) = {x0, zBf, 0, lc};
Point(6) = {x0, zWf, 0, lc};
Point(7) = {xWf, zWf, 0, lc};
Point(8) = {xWf, z0, 0, lc};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 5};
Line(15) = {5, 6};
Line(16) = {6, 7};
Line(17) = {7, 8};
Line(18) = {8, 1};
Line(19) = {6, 1};
Line(20) = {3, 6};

Line Loop(21) = {17, 18, -19, 16};
Plane Surface(22) = {21};
Line Loop(23) = {19, 11, 12, 20};
Plane Surface(24) = {23};
Line Loop(25) = {15, -20, 13, 14};
Plane Surface(26) = {25};

// following indices 1 .. 26 are seen as gmsh "Physical group tag"s (in gmsh view)
Physical Line(1) = {11};
Physical Line(4) = {14};
Physical Line(5) = {15};
Physical Line(6) = {16};
Physical Line(7) = {17};
Physical Line(8) = {18};
Physical Line(9) = {19};
Physical Line(10) = {12, 13};

Physical Surface(1) = {22};     // finally, this is subdomain_id = 1
Physical Surface(2) = {24, 26}; //                  subdomain_id = 2
