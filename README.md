# Microtubule-Wetting
Code about the microtubule wetting project.

We try to simulate the wetting of microtubule by proteins such as EBs and CLIP-170 proteins.
We use a phase field approach.
The phase separation is introduced via a double well potential (different potentials are tested, but we settled
on a simple (phi-a)²(phi-b)² type of potential).
The wetting is introduced via a surface energy h*phi.
We might try to introduce binding to the microtubule via the use of a second population evolving on the MT surface
and interacting with the bulk proteins.

First, we are simulating an infinite MT by using periodic boundary conditions along the x-axis.
The MT is at the bottom and a wall is at the top.

We expect to see a Rayleigh-Plateau instability causing the formation of regularly spaced droplets.
