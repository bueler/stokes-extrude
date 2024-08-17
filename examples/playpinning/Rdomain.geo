// demo for showing how to set subdomain ids from a .geo file
// greatly modified from https://firedrakeproject.org/demos/L_domain.geo

xl = -20.;
x0 = 0;
xr = 2.;
z0 = 0;
zt = 10.;
lc = 1.0;

// following indices are seen as gmsh "Elementary entity tag"s (in gmsh view)
Point(1) = {x0, z0, 0, lc};
Point(2) = {xr, z0, 0, lc};
Point(3) = {xr, zt, 0, lc};
Point(4) = {x0, zt, 0, lc};
Point(5) = {xl, zt, 0, lc};
Point(6) = {xl, z0, 0, lc};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(16) = {4, 5};
Line(17) = {5, 6};
Line(18) = {6, 1};
Line(19) = {4, 1};  // line along boundary between subdomains

Line Loop(21) = {17, 18, -19, 16};
Plane Surface(22) = {21};
Line Loop(23) = {19, 11, 12, 13};
Plane Surface(24) = {23};

// following indices are seen as gmsh "Physical group tag"s (in gmsh view)
Physical Line(1) = {19};

Physical Line(3) = {22}; // finally, this is subdomain_id = 1
Physical Surface(2) = {24}; //                  subdomain_id = 2
