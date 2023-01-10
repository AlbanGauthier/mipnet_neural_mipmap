###
# from https://people.math.sc.edu/Burkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid_points.py
###

#! /usr/bin/env python
#
def sphere_fibonacci_grid_points ( ng ):

#*****************************************************************************80
#
## SPHERE_FIBONACCI_GRID_POINTS: Fibonacci spiral gridpoints on a sphere.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 May 2015
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Richard Swinbank, James Purser,
#    Fibonacci grids: A novel approach to global modelling,
#    Quarterly Journal of the Royal Meteorological Society,
#    Volume 132, Number 619, July 2006 Part B, pages 1769-1793.
#
#  Parameters:
#
#    Input, integer NG, the number of points.
#
#    Output, real XG(3,N), the grid points.
#
  import numpy as np

  phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0

  theta = np.zeros ( ng )
  sphi = np.zeros ( ng )
  cphi = np.zeros ( ng )

  for i in range ( 0, ng ):
    i2 = 2 * i - ( ng - 1 ) 
    theta[i] = 2.0 * np.pi * float ( i2 ) / phi
    sphi[i] = float ( i2 ) / float ( ng )
    cphi[i] = np.sqrt ( float ( ng + i2 ) * float ( ng - i2 ) ) / float ( ng )

  xg = np.zeros ( ( ng, 3 ) )

  for i in range ( 0, ng ) :
    xg[i,0] = cphi[i] * np.sin ( theta[i] )
    xg[i,1] = cphi[i] * np.cos ( theta[i] )
    xg[i,2] = sphi[i]

  return xg