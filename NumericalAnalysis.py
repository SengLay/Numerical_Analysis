from collections.abc import Callable
import math
from typing import Optional, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmath
import sympy

# Bisection Method
def Bisection (f, x0, x1, e):
    step = 1
    condition = True
    df = pd.DataFrame(data = {'x': [x0], 'f(x)':[f(x0)]})
    while condition:
    # for i in range(10):
        x2 = (x0+x1)/2
        print(f'Step:{step}, x2 = {x2:0.16f} and f(x2) = {f(x2): 0.16f}')
        
        if f(x0) * f(x2) < 0:
            x1=x2
        else:
            x0=x2

        df.loc[step] = {'x':x0, 'f(x)':f(x0)}
        step += 1
        # condition = abs(f(x2)) > e
        condition = abs(f(x2)) < 0.0001
    print(f'\nRequied Root is: {x2:0.16f}')
    return df

# Fixed Point Method
def FixedPointMethod(p_0, funct, TOL = 1e-6, MAX_ITER = 500):
    """Solve for a function's root via Fixed-Point Method.

    Args
    ----
        p_0: Initial value of approximation
        funct (function): Function of interest for a fixed point, g(x)
        TOL: Solution tolerance
        MAX_ITER: Maximum number of iterations
    
    Return
    -------
        p: Root of p = g(p) given p_0
    """
    ## STEP 1:
    p_prev = 0.0    # Keep track of old p-values
    soln = 0.0      # Store final solution in this variable

    ## STEP 2:
    for iters in range(MAX_ITER):   # Iterate until max. iterations are reached.
        ## STEP 3:
        p = funct(p_0)             

        ## STEP 4:
        # Check if tolerance is satisfied
        if np.abs(p - p_0) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = p
            break
        
        ## STEP 5
        p_0 = p

        p_prev = p      # Replace old with new
    
    return p

# Newton's Method
def NewtonMethod(p_0, funct, Dfunct, TOL = 1e-6, MAX_ITER = 500):
    """Solve for a function's root f(x) = 0 via Newton's Method.

    Args
    ----
        p_0: Initial value of approximation
        funct (function): Function of interest, f(x)
        Dfunct : Derivative of f(x)
        TOL: Solution tolerance
        MAX_ITER: Maximum number of iterations
    
    Return
    -------
        p: Root of f(x)=0 given p_0
    """
    ## STEP 1:
    p_prev = 0.0    # Keep track of old p-values
    soln = 0.0      # Store final solution in this variable

    ## STEP 2:
    for iters in range(MAX_ITER):   # Iterate until max. iterations are reached.
        ## STEP 3:
        p = p_0 - funct(p_0)/Dfunct(p_0)             

        ## STEP 4:
        # Check if tolerance is satisfied
        if np.abs(p - p_0) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = p
            break
        
        ## STEP 5
        p_0 = p

        p_prev = p      # Replace old with new
    
    return p

# Secant Method
def Secant(f, x0, x1, e, N=100):
    step = 1
    condition = True
    df = pd.DataFrame(data = {'x0':[x0], 'x1':[x1], 'f(x1)':f(x1)})
    while condition:
        if f(x0) == f(x1):
            print('Divide by zero error!')
            break
        
        x2 = x0 - (x1 - x0) * f(x0) / (f(x1)- f(x0))
        print(f'step: {step}, x2 = {x2: 0.16f} and f(x2) = {f(x2):0.16f}')
        x0 = x1
        x1 = x2
        df.loc[step] = pd.Series(data={'x0': x0, 'x1':x1, 'f(x1)': f(x1)})
        step = step + 1
        
        if step > N:
            print('Not Conveergent!')
            break
            
        condition = abs(f(x2)) > e
    print(f'\n Required root is: %{x2:0.16f}')
    return df

# False Position Method
def FalsePositionMethod(p_0, p_1, funct, TOL = 1e-6, MAX_ITER = 500):
    """Solve for a function's root f(x) = 0 via the False Position Method.

    Args
    ----
        p_0, p_1: Initial value of approximation
        funct (function): Function of interest, f(x)
        TOL: Solution tolerance
        MAX_ITER: Maximum number of iterations
    
    Return
    -------
        p: Root of f(x)=0 given p_0, p_1
    """
    ## STEP 1:
    soln = 0.0      # Store final solution in this variable
    q_0 = funct(p_0)
    q_1 = funct(p_1)

    ## STEP 2:
    for iters in range(1, MAX_ITER):   # Iterate until max. iterations are reached.
        ## STEP 3:
        p = p_1 - q_1*(p_1 - p_0)/(q_1 - q_0)         

        ## STEP 4:
        # Check if tolerance is satisfied
        if np.abs(p - p_1) <= TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = p
            break

        ## STEP 5
        q = funct(p)
        if (np.sign(q) * np.sign(q_1)) < 0.0:
            p_0 = p_1
            q_0 = q_1

        ## STEP 6
        p_1 = p
        q_1 = q
    
    return p

# Muller Method (PolyEval + Muller For Solving)
def PolyEval(a,x, d):
    N = len(a)
    n = N-1
    v = [0]*(d+1)
    v[0] = a[n]

    for k in range(1, N,1):
        for l in range(d, 0, -1):
            v[l] = l*v[l-1] + x*v[l]
        v[0] = a[n-k] + x*v[0]

    return v 

def Muller(a, p0, p1, p2, float = 1.0e-10, maxiter= 100):
    fp0 = PolyEval(a=a, x=p0, d=0)[0]
    fp1 = PolyEval(a=a, x=p1, d=0)[0]
    fp2 = PolyEval(a=a, x=p2, d=0)[0]
    h1 = p1 -p0
    h2 = p2 - p1
    d1 = (fp1 - fp0) / h1
    d2 = (fp2 - fp1) / h2
    d = (d2 - d1) / (h2 + h1)
    df = pd.DataFrame(data={"p": [p0, p1, p2], "f(p)": [fp0, fp1, fp2]})
    i = 3
    condition = True
    while condition:
        b = d2 + h2 * d
        fp2 = PolyEval(a=a, x=p2, d=0)[0]
        D = (b**2 - 4*fp2*d)**(1/2)
        if abs(b-D) < abs(b + D):
            E = b + D
        else:
            E = b - D
        h = -2 * fp2 / E
        p = p2 + h 
        fp = PolyEval(a=a, x=p, d=0)[0]
        df.loc[i,:] = {"p": p, "f(p)": fp}
        condition = abs(h) > float
        if condition:
            p0 = p1
            p1 = p2
            p2 = p
            h1 = p1 - p0
            h2 = p2 - p1
            fp0 = PolyEval(a=a, x=p0, d=0)[0]
            fp1 = PolyEval(a=a, x=p1, d=0)[0]
            fp2 = PolyEval(a=a, x=p2, d=0)[0]
            d1 = (fp1 - fp0) / h1
            d2 = (fp2 - fp1) / h2
            d = (d2 - d1) / (h2 + h1)
            i = i + 1
    if i >= maxiter:
        print(f"Method failed after {maxiter} iterations")
    else:
        print(f"\n Root= {p:.2f}")
    return (p, df)

# Evaluate Lagrange Polynomial
def EvaluateLagrangePolynomial( i : int, x: list[complex], z: complex) -> complex:
    L = 1
    stop  = len(x)
    for j in range (0, stop, 1):
        if (i != j):
            L = L *(z-x[j])/(x[i]-x[j])
    return L

# Evaluate Newton Polynomial
def EvaluateNewtonPolynomial(a: list[complex], c: list[complex], x: complex) -> complex:
    n = len(a) - 1
    p = a[n]
    start = n-1
    for k in range (start , -1, -1):
        p = a[k] + p * (x - c[k])
    return p

# Evaluate Polynomial
def EvaluatePolynomial(a: list[complex], x: complex) -> complex:
    """
    TODO
    ----------
    Evaluate a polynomial and its first and second derivatives at a given value of `x`.

    Parameters
    ----------
    1) `a` : `list[complex]`
        list of complex coefficients `[a_0,...,a_n]` of the polynomial `p(x)=a_0+a_1 x+...+a_n x^n`
    2) `x` : `complex`
        complex value at which the polynomial to be evaluated

    Return
    ----------
    1) `p` : `complex`
        complex value of the polynomial evaluated at `x`

    Example
    ----------
    >>> import NumericalAnalysis as na
    >>> p = na.EvaluatePolynomial(a=[1, 1, 1], x=1)
    >>> print(p)
    """
    N = len(a)
    n = N - 1
    p = a[n]
    for i in range(1, N, 1):
        p = a[n - i] + x * p
    return p

# Lagrange Interpolation
def LagrangeInterpolation(x: list[complex], y:list[complex], z:complex) -> complex:
    N = len(x)
    p = 0
    for i in range(0, N, 1):
        p = p + y[i] * EvaluateLagrangePolynomial(i=i, x=x, z=z)
    return p

# Newton Divided Different
def NewtonDividedDifference(x: list[float], y: list[float]) -> tuple[list[float], pd.DataFrame]:
    """
    TODO
    ----------
    Newton form from power form using Newton Devided Difference

    Parameters
    ----------
    1) `x` : `list[float]`
        list of pairwise distinct floating values
    2) `y` : `list[float]`
        list of floating values corresponding to `x`

    Return
    ----------
    1) `b` : `list[float]`
        coefficients `[b_0,b_1,...,b_n]` of `b_0+b_1 (x-x_0)+...+b_n (x-x_0)(x-x_1)...(x-x_{n-1})`
    2) `F` : `pd.DataFrame`
        `pd.DataFrame` storing Newton Forward-Divided Difference table
    Example
    ----------
    >>> import NumericalAnalysis as na
    >>> na.pd.options.display.float_format = "{:.11f}".format
    >>> x = [1, 2, 3]
    >>> y = [2, 5, 10]
    >>> b, F = na.NewtonDividedDifference(x=x, y=y)
    >>> print(f"Newton Form: {b}")
    >>> print(F)
        Newton Form: [2, 3.0, 1.0]
            x   0             1             2
        0   1   2           NaN           NaN
        1   2   5 3.00000000000           NaN
        2   3  10 5.00000000000 1.00000000000
    """
    N = len(x)
    F = np.empty(shape=(N, N))
    F.fill(np.nan)
    F[:, 0] = y
    for i in range(1, N, 1):
        I = i + 1
        for j in range(1, I, 1):
            F[i, j] = (F[i, j - 1] - F[i - 1, j - 1]) / (x[i] - x[i - j])
    b = [F[i, i] for i in range(N)]
    F = pd.DataFrame(data=F)
    F.insert(loc=0, column="x", value=x)
    return (b, F)

# Newton Interpolation
def NewtonInterPolation(x: list[float], y: list[float], x0: float) -> tuple[float, list[float], pd.DataFrame ]:
    b, F = NewtonDividedDifference(x=x, y=y)
    p = EvaluatePolynomial(a=b, c=x[:-1], x=x0)
    return (p, b, F)
    
# Neville Interpolation
def NevilleInterpolation(x: list[float], y: list[float], x0: float) -> tuple[float, pd.DataFrame]:
    N = len(x)
    n = N - 1
    import numpy as np
    Q = np.empty(shape=(N, N))
    Q.fill(np.nan)
    Q[:,0] = y
    for i in range(1, N, 1):
        I = i + 1
        for j in range(1, I, 1):
            Q[i, j] = ((x0 - x[i-j]) * Q[i,j-1] - (x0-x[i]) * Q[i-1,j-1]) / (x[i]-x[i-j])
        p = Q[n,n]
        Q = pd.DataFrame(data=Q)
        Q.insert(loc=0, column="x", value=x)
    return(p,Q)

# Laguerre's Method Version 1
def LaguerresMethodV1(x_0, n, funct, Dfunct, DDfunct, TOL = 1e-6, MAX_ITER = 500):
    """Solve for a function's root f(x) = 0 via the Laguerre's Method.

    Args
    ----
        x_0: Initial value of approximation
        funct (function): Function of interest, f(x)
        Dfunct : First derivative of f(x), i.e. f'(x)
        DDfunct : Second derivative of f(x), i.e. f"(x)
        TOL: Solution tolerance
        MAX_ITER: Maximum number of iterations
    
    Return
    -------
        x: Root of f(x)=0 given x_0
    """
    ## STEP 1:
    soln = 0.0      # Store final solution in this variable

    ## STEP 2:
    for iters in range(MAX_ITER):   # Iterate until max. iterations are reached.
        ## STEP 3:
        p = funct(x_0)
        dp = Dfunct(x_0)
        ddp = DDfunct(x_0)

        ## STEP 4:
        # Check if tolerance is satisfied
        if np.abs(p) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = x_0
            break
        
        ## STEP 5:
        G = dp/p
        H = G**2 - ddp/p
        F = cmath.sqrt((n-1)*(n*H - G**2))
        
        ## STEP 6:
        if np.abs(G + F) > np.abs(G - F):
            a = n / (G + F)
        else:
            a = n / (G - F)
        
        ## STEP 7:
        x_0 = x_0 - a
    
        ## STEP 8:
        # Check if tolerance is satisfied
        if np.abs(a) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = x_0
            break

    return x_0

# Laguerre's Method via Polynomial Computation
def compute_polynomial(list_coeff, x):
    # Initialize result
    p = list_coeff[0]
    dp = 0
    ddp = 0
    n = len(list_coeff)
    # Evaluate value of polynomial
    for i in range(1, n):
        ddp = 2*dp + ddp*x
        dp = p + dp*x
        p = p*x + list_coeff[i]
    return p, dp, ddp

# Laguerre Method Version 2
def LaguerresMethodV2(list_coeff, x_0, TOL = 1e-6, MAX_ITER = 500):
    """Solve for a function's root f(x) = 0 via the Laguerre's Method.

    Args
    ----
        list_coeff: list of coefficients of polynomial
        x_0: Initial value of approximation
        TOL: Solution tolerance
        MAX_ITER: Maximum number of iterations
    
    Return
    -------
        x: Root of f(x)=0 given x_0
    """
    ## STEP 1:
    soln = 0.0      # Store final solution in this variable
    n = len(list_coeff)

    ## STEP 2:
    for iters in range(MAX_ITER):   # Iterate until max. iterations are reached.
        ## STEP 3:
        p, dp, ddp = compute_polynomial(list_coeff, x_0)

        ## STEP 4:
        # Check if tolerance is satisfied
        if np.abs(p) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = x_0
            break
        
        ## STEP 5:
        G = dp/p
        H = G**2 - ddp/p
        F = cmath.sqrt((n-1)*(n*H - G**2))
        
        ## STEP 6:
        if np.abs(G + F) > np.abs(G - F):
            a = n / (G + F)
        else:
            a = n / (G - F)
        
        ## STEP 7:
        x_0 = x_0 - a
    
        ## STEP 8:
        # Check if tolerance is satisfied
        if np.abs(a) < TOL:
            # Break if tolerance is met, return answer!
            print('Found solution after', iters+1, 'iterations.')
            soln = x_0
            break
            
    return x_0

# Neville Method
def Neville_Algorithm(x, y, xi):
    """
    Evaluates the interpolating polynomial that passes through the data points (x, y) at the point xi
    using Neville's algorithm.

    Parameters:
        x (list): The x-coordinates of the data points.
        y (list): The y-coordinates of the data points.
        xi (float): The point at which to evaluate the interpolating polynomial.

    Returns:
        The table Q and the value of the interpolating polynomial at the point xi.
    """
    n = len(x)
    Q = [[0] * n for i in range(n)]

    ## STEP 1:
    for i in range(n):
        Q[i][0] = y[i]

    ## STEP 2:
    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((xi - x[i+j]) * Q[i][j-1] - (xi - x[i]) * Q[i+1][j-1]) / (x[i] - x[i+j])

    return np.array(Q), Q[0][n-1]

# Forward Difference
def Forward_Difference(x, y):
    """
    This function computes the forward-difference table for the given set of data points.
    :param x: A list of x-coordinates of the data points
    :param y: A list of y-coordinates of the data points
    :return: A list of lists representing the forward-difference table
    """
    n = len(x)
    forward_table = []
    forward_table.append(y)
    for i in range(1, n):
        forward_row = []
        for j in range(n-i):
            forward_row.append(forward_table[i-1][j+1] - forward_table[i-1][j])
        forward_table.append(forward_row)
    return forward_table

# Newton Forward Interpolation 
def Newton_Forward_Interpolation(x, y, xi):
    """
    This function performs the Newton's Forward Interpolation using the forward-difference table
    computed from the above function.
    :param x: A list of x-coordinates of the data points
    :param y: A list of y-coordinates of the data points
    :param xi: The value of x for which we want to find the approximate value of y
    :return: The approximate value of y at xi
    """
    n = len(x)
    h = x[1] - x[0]
    forward_table = Forward_Difference(x, y)
    s = (xi - x[0]) / h
    result = y[0]
    for i in range(1, n):
        s_term = 1
        for j in range(i):
            s_term *= (s - j)
        result += (s_term * forward_table[i][0]) / math.factorial(i)
    return result

# Newton Forward Difference Formula
def NewtonForwardDifferenceFormula(Lx, Ly):
    if  len(Lx)!= len(Ly):
        print("The number of Lx and Ly must be the same !")
        
    n = len(Lx)
    h = Lx[1] - Lx[0]
    forward_table = Forward_Difference(Lx, Ly)
    s = sympy.symbols('s')
    result = Ly[0]
    for i in range(1, n):
        s_term = 1
        for j in range(i):
            s_term *= (s - j)
        result += (s_term * forward_table[i][0]) / math.factorial(i)
    return result

# Divided Difference
def Divided_Difference(x, y):
    """
    This function computes the divided-difference coefficients using the Newton's
    Divided-Difference formula.
    :param x: A list of x-coordinates of the data points
    :param y: A list of y-coordinates of the data points
    :return: A list of the divided-difference coefficients
    """
    n = len(x)
    coefficients = []
    for i in range(n):
        coefficients.append(y[i])
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coefficients[i] = (coefficients[i] - coefficients[i-1]) / (x[i] - x[i-j])
    return coefficients

# Newton Divided Interpolation
def Newton_Divided_Interpolation(x, y, xi):
    """
    This function performs the Newton's Interpolation using the divided-difference
    coefficients computed from the above function.
    :param x: A list of x-coordinates of the data points
    :param y: A list of y-coordinates of the data points
    :param xi: The value of x for which we want to find the approximate value of y
    :return: The approximate value of y at xi
    """
    n = len(x)
    coefficients = Divided_Difference(x, y)
    yi = coefficients[n-1]
    for i in range(n-2, -1, -1):
        yi = coefficients[i] + (xi - x[i]) * yi
    return yi

# Newton Divided Difference Formula
def NewtonDividedDifferenceFormula(Lx, Ly):
    x = sympy.symbols('x')
    if  len(Lx)!= len(Ly):
        print("The number of Lx and Ly must be the same !")
        
    n = len(Lx)
    coefficients = Divided_Difference(Lx, Ly)
    yi = coefficients[n-1]
    for i in range(n-2, -1, -1):
        yi = coefficients[i] + (x - Lx[i]) * yi
    return sympy.expand(yi)

# Reduced Row Echelon Form
def Reduced_Row_Echelon_Form(matrix):
    pivot = 0 # Tracks pivot element index
    back_swap = len(matrix)-1 # To move any rows containing zeroes to the end of the matrix
    m = len(matrix) # Number of rows
    n = len(matrix[0]) # Number of columns

    for r in range(m): # Loop through each row
        scan_r = r # scan_r allows look-ahead for rows with non-zero pivot
        while matrix[scan_r][pivot] == 0: # While the current row has a zero pivot element,
            scan_r += 1 # We increment our look-ahead row counter
            if scan_r == m: # If we hit the end of the end of the matrix, then we know that all the rows contains a zero at that pivot index.
                scan_r = r # So we start from the same row, but now we look for non-zero pivot in the next column.
                pivot += 1
                if pivot == n: # If pivot == number of columns, stop. We are done. We hit the end of the matrix.
                    break
        matrix[r], matrix[scan_r] = matrix[scan_r], matrix[r] # scan_r now has the index of non-zero pivot row. Swap raw r with scan_r row.
        pivot_element = matrix[r][pivot]  # Get  the pivot element
        for i in range(n):  # We divide all the numbers in that row by the pivot element, to make our pivot element = 1	
            matrix[r][i] /= pivot_element
        for i in range(m):# Then loop through all the other rows
            if i != r: # Ensures we don't reduce the pivot row with pivot row
                factor = matrix[i][pivot] # Factor is the number by which we need to multiply our pivot row in order to reduce row i
                for j in range(n): # Subtract all the elements in row i by factor * corresponding element in our pivot row
                    matrix[i][j] -= factor*matrix[r][j]
        pivot += 1
    return matrix

# Natural Cubic Spline
def Natural_Cubic_Spline(x, y):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    A = [0] * n # first row in A is [1, 0, 0, ...]
    A[0] = [0]*(n+1)
    A[0][0] = 1
    A[n-1] = [0] * (n+1)
    A[n-1][n-1] = 1 # Last row in A is [0, 0, 0, ......, 1]
    A[n-1][n] = 0
    idx = 0

    for i in range(1,n-1):
        a = [0] * (n+1)
        a[idx] = h[i-1]
        a[idx+1] = 2*(h[i-1] + h[i])
        a[idx+2] = h[i]
        a[n] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        idx += 1
        A[i] = a
    reduced = Reduced_Row_Echelon_Form(A) # Reduce the augmented matrix A|b to get the c in the last column

    b = [0] * (n-1)
    c = [0] * (n-1)
    d = [0] * (n-1)
    for i in range(n-1):
        c[i] = reduced[i][-1]
        c2 = reduced[i+1][-1]
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(c2 + 2*c[i])/3 # Compute b and d
        d[i] = (c2-c[i])/(3*h[i]) 

    for i in range(n-1): # Print out each equation if requested.
        factor = f"(x{')' if x[i] == 0 else f' - {x[i]:.2f})' }"
        print(f"S_{i}(x) = ", end='')
        if y[i] != 0:
            if y[i] < 0:
                print("-", end ='')
            print("%5.4f" % abs(y[i]), end = ' ')
        if b[i] != 0:
            print("-" if b[i] < 0 else "+", end =' ')
            print("%5.4f%s" % (abs(b[i]), factor), end = ' ')
        if c[i] != 0:
            print("-" if c[i] < 0 else "+", end =' ')
            print("%5.4f%s^2" % (abs(c[i]), factor), end = ' ')
        if d[i] != 0:
            print("-" if d[i] < 0 else "+", end =' ')
            print("%5.4f%s^3" % (abs(d[i]), factor))
        print("\n")
    return [y[:(n-1)], b, c, d]

# Natual Cubic Spline (Course Algoithm)
# zero vectors
def zeroV(m):
    z = [0]*m
    return(z)

#INPUT: n; x0, x1, ... ,xn; a0 = f(x0), a1 =f(x1), ... , an = f(xn).
def Natural_Cubic_Spline_Course(x, y):
    """
        This function interpolates between the knots
        specified by lists x and y. The function computes the coefficients
        and outputs the solution of possible S(x).
    """          

    n = len(x)
    a = y
    h = zeroV(n-1)

    # alpha will be values in a system of eq's that will allow us to solve for c
    # and then from there we can find b, d through substitution.
    alpha = zeroV(n-1)

    # l, u, z are used in the method for solving the linear system
    l = zeroV(n+1)
    u = zeroV(n)
    z = zeroV(n+1)

    # b, c, d will be the coefficients along with a.
    b = zeroV(n)     
    c = zeroV(n+1)
    d = zeroV(n)    

    for i in range(n-1):
        # h[i] is used to satisfy the condition that 
        # Si+1(xi+l) = Si(xi+l) for each i = 0,..,n-1
        # i.e., the values at the knots are "doubled up"
        h[i] = x[i+1]-x[i]  

    for i in range(1, n-1):
        # Sets up the linear system and allows us to find c.  Once we have 
        # c then b and d follow in terms of it.
        alpha[i] = (3./h[i])*(a[i+1]-a[i])-(3./h[i-1])*(a[i] - a[i-1])

    # I, II, (part of) III Sets up and solves tridiagonal linear system...
    # I   
    l[0] = 1      
    u[0] = 0      
    z[0] = 0

    # II
    for i in range(1, n-1):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

    l[n] = 1
    z[n] = 0
    c[n] = 0

    # III... also find b, d in terms of c.
    for j in range(n-2, -1, -1):      
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3.
        d[j] = (c[j+1] - c[j])/(3*h[j]) 
        
    for i in range(n-1): # Print out each equation if requested.
        factor = f"(x{')' if x[i] == 0 else f' - {x[i]:.2f})' }"
        print(f"S_{i}(x) = ", end='')
        if y[i] != 0:
            if y[i] < 0:
                print("-", end ='')
            print("%5.4f" % abs(y[i]), end = ' ')
        if b[i] != 0:
            print("-" if b[i] < 0 else "+", end =' ')
            print("%5.4f%s" % (abs(b[i]), factor), end = ' ')
        if c[i] != 0:
            print("-" if c[i] < 0 else "+", end =' ')
            print("%5.4f%s^2" % (abs(c[i]), factor), end = ' ')
        if d[i] != 0:
            print("-" if d[i] < 0 else "+", end =' ')
            print("%5.4f%s^3" % (abs(d[i]), factor))
        print("\n")
        
    return [y[:(n-1)], b[:(n-1)], c[:(n-1)], d[:(n-1)]]

# Clamped Cubic Spline
#INPUT: n; x0, x1, ... ,xn; a0 = f(x0), a1 =f(x1), ... , an = f(xn), FPO=f'(x0), FPN=f'(xn).
def Clamped_Cubic_Spline_Course(x, y, FPO, FPN):
    """
        This function interpolates between the knots
        specified by lists x and y. The function computes the coefficients
        and outputs the solution of possible S(x).
    """        
    
    n = len(x)
    a = y
    h = zeroV(n-1)

    # alpha will be values in a system of eq's that will allow us to solve for c
    # and then from there we can find b, d through substitution.
    alpha = zeroV(n)

    # l, u, z are used in the method for solving the linear system
    l = zeroV(n+1)
    u = zeroV(n)
    z = zeroV(n+1)

    # b, c, d will be the coefficients along with a.
    b = zeroV(n)     
    c = zeroV(n+1)
    d = zeroV(n)    

    for i in range(n-1):
        # h[i] is used to satisfy the condition that 
        # Si+1(xi+l) = Si(xi+l) for each i = 0,..,n-1
        # i.e., the values at the knots are "doubled up"
        h[i] = x[i+1]-x[i]  
        
    alpha[0] = 3*(a[1] - a[0])/h[0] - 3*FPO
    alpha[n-1] = 3*FPN - 3*(a[n-1] - a[n-2])/h[n-2]

    for i in range(1, n-1):
        # Sets up the linear system and allows us to find c.  Once we have 
        # c then b and d follow in terms of it.
        alpha[i] = (3./h[i])*(a[i+1]-a[i])-(3./h[i-1])*(a[i] - a[i-1])

    # I, II, (part of) III Sets up and solves tridiagonal linear system...
    # I   
    l[0] = 2*h[0]     
    u[0] = 0.5      
    z[0] = alpha[0]/l[0]

    # II
    for i in range(1, n-1):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

    l[n] = h[n-2]*(2-u[n-2])
    z[n] = (alpha[n-1] - h[n-2]*z[n-2])/l[n]
    c[n] = z[n]

    # III... also find b, d in terms of c.
    for j in range(n-2, -1, -1):      
        c[j] = z[j] - u[j]*c[j+1]
        b[j] = (a[j+1] - a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3.
        d[j] = (c[j+1] - c[j])/(3*h[j]) 
        
    for i in range(n-1): # Print out each equation if requested.
        factor = f"(x{')' if x[i] == 0 else f' - {x[i]:.2f})' }"
        print(f"s_{i}(x) = ", end='')
        if y[i] != 0:
            if y[i] < 0:
                print("-", end ='')
            print("%5.4f" % abs(y[i]), end = ' ')
        if b[i] != 0:
            print("-" if b[i] < 0 else "+", end =' ')
            print("%5.4f%s" % (abs(b[i]), factor), end = ' ')
        if c[i] != 0:
            print("-" if c[i] < 0 else "+", end =' ')
            print("%5.4f%s^2" % (abs(c[i]), factor), end = ' ')
        if d[i] != 0:
            print("-" if d[i] < 0 else "+", end =' ')
            print("%5.4f%s^3" % (abs(d[i]), factor))
        print("\n")
        
    return [y[:(n-1)], b[:(n-1)], c[:(n-1)], d[:(n-1)]]

# Richardson Extrapolation
def Richardson_Extrapolation(f: Callable[[float], float],
                            x: float,
                            h: float,
                            n: int = 9,
                            rtol: float = 1e-10
                            ) -> tuple[float, pd.DataFrame]:
    N = n + 1
    D = np.full(shape=(N,N), fill_value=np.nan, dtype=np.float64)
    D[0, 0] = 0.5 * (f(x + h) - f(x - h)) / h
    for i in range(1, N, 1):
        h = 0.5 * h
        D[i, 0] = 0.5 * (f(x + h) - f(x - h)) / h
        I = i + 1
        p = 1
        for j in range(1, I, 1):
            p = 4*p
            D[i, j] = D[i, j-1] + (D[i, j-1] - D[i-1, j-1]) / (p-1)
        if abs(D[i,i] - D[i-1, i-1]) < rtol:
            break
    d = D[i,i]
    columns = [f'O(h^{2*(k+1)})' for k in range(0, I, 1)]
    D = pd.DataFrame(data=D[:I, :I], columns=columns)
    return (d, D)

# Trapezoidal Rule
def TrapezoidalRule(f, a, b):
    h = b - a
    return (h * 0.5 * (f(a) + f(b)))

# Simpson Rule
def SimpsonRule(f, a, b):
    h = (b - a)/2
    c = a + h
    return (h/3 * (f(a) + 4*f(c) + f(b))) 

# Midpoint Rule
def MidpointRule(f, a, b):
    h = (b - a)/2
    x_0 = a + h
    return 2*h*f(x_0)

# Composite Trapezoidal
def CompositeTrapezoidal(f: Callable[[float], float], a:float, b:float, n:int) -> float:
    """
    --------------------
    Approximate the integral of 'f(x)' from 'a' to 'b' using Composite Trapezoidal rule/formula
    
    Parameters
    --------------------
    'f' : 'callable'
        function that we want to integrate
    'a' : 'float'
        lower limit of the integral
    'b' : 'float'
        upper limit of the integral
    'n' : 'int'
        (n+1) be the number of nodes
        
    Return
    -------------------
    'A' : 'float'
        the approximated value of the integral
    """
    h = (b-a) / n
    f_0 = f(a) + f(b)
    f_i = 0
    for i in range(1, n):
        x = a + i*h
        f_i = f_i + f(x)
    A = 0.5 * h * (f_0 + 2*f_i)
    return A

# Composite Simpson
def CompositeSimpson(f: Callable[[float], float], a:float, b:float, n:int) -> float:
    """
    --------------------
    Approximate the integral of 'f(x)' from 'a' to 'b' using Composite Simpson rule/formula
    
    Parameters
    --------------------
    'f' : 'callable'
        function that we want to integrate
    'a' : 'float'
        lower limit of the integral
    'b' : 'float'
        upper limit of the integral
    'n' : 'int'
        (n+1) be the number of nodes
        
    Return
    -------------------
    'A' : 'float'
        the approximated value of the integral
    """
    h = (b-a) / n
    f_0 = f(a) + f(b)
    f_1 = 0
    f_2 = 0
    for i in range(1, n):
        x = a + i*h
        if (i % 2 == 0):
            f_2 = f_2 + f(x)
        else:
            f_1 = f_1 + f(x)
    A = h * (f_0 + 2*f_2 + 4*f_1) / 3
    return A

# Composite Midpoint
def CompositeMidpoint(f: Callable[[float], float], a:float, b:float, n:int) -> float:
    """
    --------------------
    Approximate the integral of 'f(x)' from 'a' to 'b' using Composite Midpoint rule/formula
    
    Parameters
    --------------------
    'f' : 'callable'
        function that we want to integrate
    'a' : 'float'
        lower limit of the integral
    'b' : 'float'
        upper limit of the integral
    'n' : 'int'
        (n+1) be the number of nodes
        
    Return
    -------------------
    'A' : 'float'
        the approximated value of the integral
    """
    h = (b-a) / (n+2)
    f_2 = 0
    for i in range(0, n+3):
        x = a + i*h
        if (i % 2 != 0):
            f_2 = f_2 + f(x)
    A = 2 * h * f_2
    return A

# Romberg Integration
def Romberg_Integration(f, a, b, n):
    """Calculate the integral from the Romberg method.
    Args:
        f (function): the equation f(x).
        a (float): the initial point.
        b (float): the final point.
        n (int): number of intervals.
    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    # Initialize the Romberg integration table
    R = np.zeros((n, n))

    # Compute the trapezoid rule for the first column (h = b - a)
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))

    # Iterate for each level of refinement
    for i in range(1, n):
        h = 0.5 * h  # Halve the step size
        # Compute the composite trapezoid rule
        sum_f = 0
        for j in range(1, 2**i, 2):
            x = a + j * h
            sum_f += f(x)
        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_f

        # Richardson extrapolation for higher order approximations
        for k in range(1, i + 1):
            R[i, k] = R[i, k - 1] + \
                (R[i, k - 1] - R[i - 1, k - 1]) / ((4**k) - 1)
            
    columns = [f'O(h^{2*(k+1)})' for k in range(0, n, 1)]
    D = pd.DataFrame(data=R[:n, :n], columns=columns)

    #return float(R[n - 1, n - 1])
    return D

# Adaptive Simpson Quadrature
def Adaptive_Simpson(f, a, b, tol=1e-10):
    """
    Evaluates the integral of f(x) on [a,b].

    USAGE:
        s = adaptive_simpson( f, a, b, tol )

    INPUT:
        f         - function to integrate
        a, b      - left and right endpoints of interval of integration
        tol       - desired upper bound on allowable error

    OUTPUT:
        float     - the value of the integral

    NOTES:
        Integrates the function f(x) on [a,b] with an error bound of
        given by tol using an adaptive Simpson's rule to approximate
        the value of the integral.  Notice that this is not very
        efficient -- it is recursive and recomputes many function
        values that have already been computed.  The code in this
        function is meant to be clear and explain the ideas behind
        adaptive schemes rather than to be an efficient implementation
        of one.

    # Theory says the factor to multiply the tolerance by should be 15, but
    # since that assumes that the fourth derivative of f is fairly constant,
    # we want to be a bit more conservative...
    """

    tol_factor = 10.0

    h = 0.5 * ( b - a )

    x0 = a
    x1 = a + 0.5 * h
    x2 = a + h
    x3 = a + 1.5 * h
    x4 = b

    f0 = f( x0 )
    f1 = f( x1 )
    f2 = f( x2 )
    f3 = f( x3 )
    f4 = f( x4 )

    s0 = h * ( f0 + 4.0 * f2 + f4 ) / 3.0
    s1 = h * ( f0 + 4.0 * f1 + 2.0 * f2 + 4.0 * f3 + f4 ) / 6.0

    if abs( s0 - s1 ) >= tol_factor * tol:
        s = Adaptive_Simpson( f, x0, x2, 0.5 * tol ) + \
            Adaptive_Simpson( f, x2, x4, 0.5 * tol )
    else:
        s = s1 + ( s1 - s0 ) / 15.0

    return s

# Guassian Quadrature
def Gaussian_Quadrature(f: Callable[[float], float], a:float, b:float, n:int) -> float:
    """
    --------------------
    Approximate the integral of 'f(x)' from 'a' to 'b' using Gaussian quadrature for n=1,2,3,4,5 (only)
    
    Parameters
    --------------------
    'f' : 'callable'
        function that we want to integrate
    'a' : 'float'
        lower limit of the integral
    'b' : 'float'
        upper limit of the integral
    'n' : 'int'
        (n+1) be the number of nodes
        
    Return
    -------------------
    'A' : 'float'
        the approximated value of the integral
    """
    c_2 = 1.0
    x_2_1 = 0.5773502692
    x_2_2 = -0.5773502692
    
    c_3_1 = 0.8888888889
    c_3_2 = 0.5555555556
    c_3_3 = 0.5555555556
    x_3_1 = -0.0
    x_3_2 = -0.7745966692
    x_3_3 = 0.7745966692
    
    c_4_1 = 0.6521451549
    c_4_2 = 0.6521451549
    c_4_3 = 0.3478548451
    c_4_4 = 0.3478548451
    x_4_1 = 0.3399810436
    x_4_2 = -0.3399810436
    x_4_3 = -0.8611363116
    x_4_4 = 0.8611363116
    
    c_5_1 = 0.5688888889
    c_5_2 = 0.4786286705
    c_5_3 = 0.4786286705
    c_5_4 = 0.2369268851
    c_5_5 = 0.2369268851
    x_5_1 = -0.0000000000
    x_5_2 = 0.5384693101 
    x_5_3 = -0.5384693101
    x_5_4 = -0.9061798459
    x_5_5 = 0.9061798459
    
    A = 0.0
    if (n == 2):
        A = c_2*0.5*(b-a)*(f(0.5*(b+a+(b-a)*x_2_1)) + f(0.5*(b+a+(b-a)*x_2_2)))
    elif (n == 3):
        A = 0.5*(b-a)*(c_3_1*f(0.5*(b+a+(b-a)*x_3_1)) + c_3_2*f(0.5*(b+a+(b-a)*x_3_2)) + c_3_3*f(0.5*(b+a+(b-a)*x_3_3)))
    elif (n == 4):
        A = 0.5*(b-a)*(c_4_1*f(0.5*(b+a+(b-a)*x_4_1)) + c_4_2*f(0.5*(b+a+(b-a)*x_4_2)) + c_4_3*f(0.5*(b+a+(b-a)*x_4_3)) + c_4_4*f(0.5*(b+a+(b-a)*x_4_4)))
    elif (n == 5):
        A = 0.5*(b-a)*(c_5_1*f(0.5*(b+a+(b-a)*x_5_1)) + c_5_2*f(0.5*(b+a+(b-a)*x_5_2)) + c_5_3*f(0.5*(b+a+(b-a)*x_5_3)) + c_5_4*f(0.5*(b+a+(b-a)*x_5_4)) + c_5_5*f(0.5*(b+a+(b-a)*x_5_5))) 
    else:
        print("The solution for n=",n,"is not yet defined !!!")
    
    return A

# Forward Substitution
def ForwardSubstitution(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    
    """
    Solve for Y from form of LY = B

    Params
    -------------------
    `L`: Lower Triangonal Matrix
        np.ndarray
    `B`: Vector
        np.ndarray

    Return
    -------------------
    `Y`: Vector found from form of LY = B
        np.ndarray
    """
    
    N = L.shape[0]
    X = np.zeros(N)
    X[0] = B[0] / L[0,0]

    for i in range(0, N):
        X[i] = B[i]
        start = i - 1
        for j in range(start, -1, -1):
            X[i] -= L[i,j] * X[j]
        X[i] = X[i] / L[i,i]
        
    return X

# Backward Substitution
def BackwardSubstitution_TP(U, Y):
    
    """
    Solve for X from form of UX = Y

    Params
    -------------------
    `U`: Upper Triangonal Matrix
        np.ndarray
    `Y`: Vector
        np.ndarray

    Return
    -------------------
    `X`: Roots as Vector from form UX = Y
        np.ndarray
    """
    
    N = U.shape[0]
    X = np.zeros(N)
    X[N-1] = Y[N-1] / U[N-1, N-1]
    
    start_i = N - 2
    for i in range(start_i, -1, -1):
        X[i] = Y[i]
        start_j = i + 1
        for j in range(start_j, N):
            X[i] -= U[i,j] * X[j]
        X[i] /= U[i,i]
        
    return X

# Backward Substition
def BackwardSubstitution_Course(U: np.ndarray,
                        Y: np.ndarray
                        ) -> np.ndarray: 
    
    """
    Solve for X from form of UX = Y

    Params
    -------------------
    `U`: Upper Triangonal Matrix
        np.ndarray
    `Y`: Vector
        np.ndarray

    Return
    -------------------
    `X`: Roots as Vector from form UX = Y
        np.ndarray
    """
    
    N = U.shape[0]
    X = np.zeros_like(Y)
    X[N - 1] = Y[N - 1] / U[N - 1, N - 1]

    start_i = N - 2
    for i in range(start_i, -1, -1):
        X[i] = Y[i]
        start_j = i + 1
        for j in range(start_j, N, 1):
            X[i] -= U[i, j] * X[j]
        X[i] /= U[i, i]

    return X

# Guass Elimination
def GuassElimination(a: np.ndarray,
                    b: np.ndarray) -> tuple[np.ndarray,
                                            np.ndarray]:
    _a = a.copy()
    _b = b.copy()
    n = _a.shape[0]
    stop = n - 1

    for k in range(0, stop, 1):
        start = k + 1
        for i in range(start, n, 1):
            r = _a[i, k] / _a[k, k]
            _a[i, k] = 0
            for j in range(start, n, 1):
                _a[i, j] = _a[i, j] - r * _a[k, j]
            _b[i] = _b[i] - r * _b[k]

    return (_a, _b)

# DooLittle Decomposition
def DooLittleDecomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Solve for L(Lower Triangonal Matrix) & U(Upper Triangonal Matrix) using DooLittle Decomposition.
    
    Params
    ----------------------------------------
    `A`: Squared Matrix for Decomposition
        np.ndarray
        
    Return
    ----------------------------------------
    `L`: Lower Triangonal Matrix
        np.ndarray
    `U`: Upper Triangonal Matrix
        np.ndarray
    """
    
    N = A.shape[0]
    U = A.copy()
    L = np.eye(N)
    
    for k in range(0, N-1):
        start = k + 1
        for i in range(start, N):
            r = U[i, k] / U[k, k]
            L[i, k] = r
            U[i, k] = 0
            for j in range(start, N):
                U[i, j] = U[i, j] - r * U[k, j]
                
    return (L, U)

# DooLittle LDLt Decomposition
def DoolittleLDLtDecomposition(a: np.ndarray)-> tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    
    """
    Doolittle LDLt Decomposition uses to solve for 
        L(Lower Triangonal Matrix), 
        D(Diagonal Matrix) & 
        Lt(Tranpose of Lower Triangonal Matrix).
    
    Params
    --------------------
    `a`: Squared Matrix with shape of NxN
        np.ndarray
    
    Returns
    --------------------
    `l`: Lower Triangonal Matrix
        np.ndarray
    `d`: Diagonal Matrix
        np.ndarray
    `l.transpose()`: Transpose Matrix of Lower Triangonal Matrix
        np.ndarray
    """
    
    N = a.shape[0]
    l = np.eye(N=N)
    d = np.zeros(shape=N)
    
    d[0] = a[0,0]
    l[1,0] = a[0,0]
    d[1] = a[1,1] - d[0] * l[1,0]**2
    
    for i in range(2,N,1):
        l[i,0] = a[i,0]/d[0]
        for j in range(1,i,1):
            l[i,j] = a[i,j]
            for k in range(0,j,1):
                l[i,j] = l[i,j] - d[k] * l[i,k]*l[j,k]
            l[i,j] = l[i,j] / d[j]
        d[i] = a[i,i]
        for k in range (0,i,1):
            d[i] = d[i] - d[k] * l[i,k]**2
            
    return (l, d, l.transpose())

# Cholesky LLt Decomposition
def CholeskyLLtDecomposition(A: np.ndarray) -> tuple[np.ndarray, 
                                                    np.ndarray]:
    
    """
    Cholesky LLt Decomposition uses to solve for 
        L(Lower Triangonal Matrix) &
        Lt(Tranpose of Lower Triangonal Matrix).
    
    Params
    --------------------
    `A`: Squared Matrix with shape of NxN
        np.ndarray
    
    Returns
    --------------------
    `l`: Lower Triangonal Matrix
        np.ndarray
    `l.transpose()`: Transpose Matrix of Lower Triangonal Matrix
        np.ndarray
    """
    
    N = A.shape[0]
    l = np.eye(N=N)
    
    l[0,0] = np.sqrt(A[0,0])
    l[1,0] = A[1,0] / l[0,0]
    l[1,1] = np.sqrt(A[1,1] - l[1,0]**2)
    
    for i in range(2, N):
        l[i,0] = A[i,0] / l[0,0]
        for j in range(1, i):
            factor_1 = 0
            for k in range(0,j,1):
                factor_1 += l[i,k]*l[j,k]
            l[i,j] = (A[i,j] - factor_1) / l[j,j]
            
        factor_2 = 0
        for k in range(0,i,1):
            factor_2 += l[i,k] * l[i,k]
        l[i,i] = np.sqrt(A[i,i] - factor_2)

    return (l, l.transpose())

def tridiagonal_lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)
    U[0, 0] = A[0, 0]
    U[0, 1] = A[0, 1]

    for i in range(1, n-1):
        L[i, i-1] = A[i, i-1] / U[i-1, i-1]
        U[i, i] = A[i, i] - L[i, i-1] * A[i-1, i]
        U[i, i+1] = A[i, i+1]

    L[n-1, n-2] = A[n-1, n-2] / U[n-2, n-2]
    U[n-1, n-1] = A[n-1, n-1] - L[n-1, n-2] * A[n-2, n-1]

    return L, U

def PentadiagonalDecomposition(d, e, f):
    """
    LU decomposition of symetric pentadiagonal matrix [a], where
    {f}, {e} and {d} are the diagonals of [a]. 
    Return: {d},{e} and {f} : the diagonals of the decomposed matrix.
    """
    n = len(d)
    for k in range(n-2):
        lam = e[k]/d[k]
        d[k+1] = d[k+1] - lam*e[k]
        e[k+1] = e[k+1] - lam*f[k]
        e[k] = lam
        lam = f[k]/d[k]
        d[k+2] = d[k+2] - lam*f[k]
        f[k] = lam
    lam = e[n-2]/d[n-2]
    d[n-1] = d[n-1] - lam*e[n-2]
    e[n-2] = lam
    return (d, e, f)

def PentadiagonalSolve(d, e, f, B):
    n = len(d)
    B[1] = B[1] - e[0]*B[0]
    for k in range(2,n):
        B[k] = B[k] - e[k-1]*B[k-1] - f[k-2]*B[k-2]
        
    B[n-1] = B[n-1]/d[n-1]
    B[n-2] = B[n-2]/d[n-2] - e[n-2]*B[n-1]
    for k in range(n-3,-1,-1):
        B[k] = B[k]/d[k] - e[k]*B[k+1] - f[k]*B[k+2]
    return B

def gaussian_scaled_row_pivoting(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    s = np.zeros(n)
    p = np.arange(n)  # Pivot array
    
    # Compute scale factors
    for i in range(n):
        s[i] = max(abs(A[i, :]))
    
    for k in range(n-1):
        # Find pivot row
        pivot_row = np.argmax(abs(A[k:n, k]) / s[k:n]) + k
        
        # Swap rows in A and b
        A[[k, pivot_row], :] = A[[pivot_row, k], :]
        b[[k, pivot_row]] = b[[pivot_row, k]]
        
        # Swap pivot array
        p[[k, pivot_row]] = p[[pivot_row, k]]
        
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k] = factor
            for j in range(k+1, n):
                A[i, j] = A[i, j] - factor * A[k, j]
            b[i] = b[i] - factor * b[k]
    
    # Back substitution
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_val = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - sum_val) / A[i, i]
    
    return x, p

def Jacobi_iteration(A, b, x_0, MAX_ITER=1000, TOL=1e-6):
    n = A.shape[0]
    x = np.copy(x_0)
    nb_iters = 0
    
    T = A - np.diag(np.diagonal(A))
    
    for iteration in range(MAX_ITER):
        x_old  = x.copy()
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        
        if np.linalg.norm(x_old - x) < TOL:
            break
        
        nb_iters = iteration
        
    print(nb_iters)
    return x

def Gauss_Seidel_iteration(A, b, x_0, MAX_ITER=1000, TOL=1e-6):
    n = A.shape[0]
    x = np.copy(x_0)
    nb_iters = 0
    for iteration in range(MAX_ITER):
        x_new = np.zeros_like(x)
        
        for i in range(n):
            sum_val = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum_val) / A[i, i]
        
        if np.linalg.norm(x_new - x) < TOL:
            break
        
        x = x_new
        nb_iters = iteration
        
    print(nb_iters)
    return x

def Strictly_Diagonally_Dominant(A):
    D = np.diag(np.abs(A)) # Find diagonal coefficients
    S = np.sum(np.abs(A), axis=1) - D # Find row sum without diagonal
    if np.all(D > S):
        print('This matrix is diagonally dominant.')
    else:
        print('NOT diagonally dominant!')
        
def Relaxation_iteration(A, b, x_0, omega, MAX_ITER=1000, TOL=1e-6):
    n = A.shape[0]
    x = np.copy(x_0)

    for iteration in range(MAX_ITER):
        x_new = np.copy(x)

        for i in range(n):
            sum_val = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sum_val)

        if np.linalg.norm(x_new - x) < TOL:
            break

        x = x_new
        nb_iters = iteration
        
    print(nb_iters)

    return x

def Conjugate_Gradient_method(A, b, x_0, MAX_ITER=1000, TOL=1e-6):
    n = A.shape[0]
    x = np.copy(x_0)
    r = b - A @ x
    p = np.copy(r)
    rs_old = np.dot(r, r)

    for iteration in range(MAX_ITER):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < TOL:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

### TP 5
# Forward Euler Method
def ForwardEulerMethod(f, a, b, N, alpha):
    t_list = []
    w_list = []
    y_list = []
    
    h = (b - a) / N
    t = a
    w = alpha
    
    t_list.append(t)
    w_list.append(w)
    y_list.append((t + 1)**2 - (np.exp(t)/2))

    for i in range(1, N+1):
        w = w + h * f(t, w)
        t = a + i * h
        y = (t + 1)**2 - (np.exp(t)/2)
        
        t_list.append(t)
        w_list.append(w)
        y_list.append(y)
        
        df = pd.DataFrame({"t": t_list, "w": w_list, "y": y_list}, columns=['t', 'w', 'y'])
        df.loc[:, "|y-w|"] = abs(df.loc[:, "y"] - df.loc[:, "w"])
        
        pd.options.display.float_format = "{:.8f}".format
        
    return df

def taylor_method_order_2(f, df_dt, df_dy, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        t_i = t[i-1]
        y_i = y[i-1]
        k1 = f(t_i, y_i)
        if df_dy == 0:
            k2 = df_dt(t_i, y_i)
        else:
            k2 = (df_dt(t_i, y_i) + df_dy(t_i, y_i) * f(t_i, y_i))
        y[i] = y_i + h*k1 + 0.5*h*h*k2
    
    return t, y

def Explicit_midpoint_method_V1(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_1)
        y[i] = y[i-1] + h * k_2
    
    return (t, y)

def Explicit_midpoint_method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_1)
        y[i] = y[i-1] + h * k_2
    
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Modified_Euler_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + h, y[i-1] + h*k_1)
        y[i] = y[i-1] + 0.5*h * (k_1 + k_2)
        
    return (t, y)

def Heun_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + h, y[i-1] + h*k_1)
        y[i] = y[i-1] + 0.5*h * (k_1 + k_2)
        
    return (t, y)

def Ralston_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + (2/3)*h, y[i-1] + (2/3)*h*k_1)
        y[i] = y[i-1] + 0.25*h*k_1 + (3/4)*h*k_2
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Runge_Kutta_Third_Order_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_1)
        k_3 = f(t[i-1] + h, y[i-1] + h*(-k_1 + 2*k_2))
        y[i] = y[i-1] + h*((1/6)*k_1 + (2/3)*k_2 + (1/6)*k_3)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Heun_Third_Order_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + (1/3)*h, y[i-1] + (1/3)*h*k_1)
        k_3 = f(t[i-1] + (2/3)*h, y[i-1] + (2/3)*h*k_2)
        y[i] = y[i-1] + h*((1/4)*k_1 + (3/4)*k_3)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Ralston_Third_Order_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_1)
        k_3 = f(t[i-1] + (3/4)*h, y[i-1] + (3/4)*h*k_2)
        y[i] = y[i-1] + h*((2/9)*k_1 + (1/3)*k_2 + (4/9)*k_3)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Wray_VanDerHouwen_Third_Order_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + (8/15)*h, y[i-1] + (8/15)*h*k_1)
        k_3 = f(t[i-1] + (2/3)*h, y[i-1] + (1/4)*h*k_1 + (5/12)*h*k_2)
        y[i] = y[i-1] + h*((1/4)*k_1 + (3/4)*k_3)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df


def SSPRK3_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + h, y[i-1] + h*k_1)
        k_3 = f(t[i-1] + 0.5*h, y[i-1] + 0.25*h*(k_1 + k_2))
        y[i] = y[i-1] + h*((1/6)*k_1 + (1/6)*k_2 + (2/3)*k_3)
    
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])    
    return df

# Runge Kutta Method of Order 4
def Original_Runge_Kutta_Method_V2(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_1)
        k_3 = f(t[i-1] + 0.5*h, y[i-1] + 0.5*h*k_2)
        k_4 = f(t[i-1] + h, y[i-1] + h*k_3)
        y[i] = y[i-1] + h*((1/6)*k_1 + (1/3)*k_2 + (1/3)*k_3 + (1/6)*k_4)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])    
    return df

def Original_Runge_Kutta_Method(f, t_0, t_f, y_0, h):
    # Implement the original Runge-Kutta method here and return 't_' and 'y_' arrays.
    # Make sure 'y_' is a numerical array containing the solution at different time points.
    # Example: 
    t_ = np.arange(t_0, t_f + h, h)
    y_ = np.zeros(len(t_))
    y_[0] = y_0
    for i in range(len(t_) - 1):
        K1 = h * f(t_[i], y_[i])
        K2 = h * f(t_[i] + h/2, y_[i] + K1/2)
        K3 = h * f(t_[i] + h/2, y_[i] + K2/2)
        K4 = h * f(t_[i] + h, y_[i] + K3)
        y_[i + 1] = y_[i] + (K1 + 2*K2 + 2*K3 + K4)/6
    return t_, y_

def Three_Eight_Rule_Fourth_Order_Method(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1])
        k_2 = f(t[i-1] + (1/3)*h, y[i-1] + (1/3)*h*k_1)
        k_3 = f(t[i-1] + (2/3)*h, y[i-1] + h*((-1/3)*k_1 + k_2))
        k_4 = f(t[i-1] + h, y[i-1] + h*(k_1 - k_2 + k_3))
        y[i] = y[i-1] + h*((1/8)*k_1 + (3/8)*k_2 + (3/8)*k_3 + (1/8)*k_4)
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])    
    return df

def Runge_Kutta_Fehlberg(f, t_0, t_f, y_0, h_min, h_max, TOL=1e-6):
    """Runge-Kutta-Fehlberg method to solve y' = f(x,t) with y(t[0]) = y_0.

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        t_0     - left-hand endpoint of interval (initial condition is here)
        t_f     - right-hand endpoint of interval
        y_0    - initial x value: x0 = x(a)
        TOL   - maximum value of local truncation error estimate
        h_max  - maximum step size
        h_min  - minimum step size

    OUTPUT:
        t     - NumPy array of independent variable values
        y     - NumPy array of corresponding solution function values
    """

    # Coefficients used to compute the independent variable argument of f

    c2  =   1/4
    c3  =   3/8
    c4  =   12/13
    c5  =   1
    c6  =   1/2

    # Coefficients used to compute the dependent variable argument of f

    a21 =   1/4
    a31 =   3/32
    a32 =   9/32
    a41 =   1932/2197
    a42 =   -7200/2197
    a43 =   7296/2197
    a51 =   439/216
    a52 =   -8
    a53 =   3680/513
    a54 =  -845/4104
    a61 =  -8/27
    a62 =   2
    a63 =  -3544/2565
    a64 =   1859/4104
    a65 =  -11/40

    # Coefficients used to compute 4th order RK estimate

    b1  =   25/216
    b3  =   1408/2565
    b4  =   2197/4104
    b5  =  -1/5
    b6 = 0
    
    b1_star  =   16/135
    b3_star  =   6656/12825
    b4_star  =   28561/56430
    b5_star  =  -9/50
    b6_star  =   2/55
    
    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1  =   b1_star - b1
    r3  =   b3_star - b3
    r4  =   b4_star - b4
    r5  =   b5_star - b5
    r6  =   b6_star - b6

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
  
    t = t_0
    y = y_0
    h = h_max

    # Initialize arrays that will be returned

    T = np.array( [t] )
    Y = np.array( [y] )

    while t < t_f:

        # Adjust step size when we get to last interval

        if t + h > t_f:
            h = t_f - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.

        k1 = h * f(t, y)
        k2 = h * f(t + c2 * h, y + a21 * k1)
        k3 = h * f(t + c3 * h, y + a31 * k1 + a32 * k2)
        k4 = h * f(t + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)
        k5 = h * f(t + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k6 = h * f(t + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
    
        R = np.abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        if len( np.shape(R) ) > 0:
            r = np.max(R)
        if R <= TOL:
            t = t + h
            y = y + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5
            T = np.append(T, t)
            Y = np.append(Y, [y], 0)

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( TOL / R )**0.25, 0.1 ), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            print( "Error: stepsize should be smaller than %e." % h_min )
            break

    # endwhile

    return (T, Y)

def Runge_Kutta_Method(f, t_0, t_f, y_0, h_min, h_max, TOL=1e-6, method="Fehlberg"):
    """Runge-Kutta method to solve y' = f(x,t) with y(t[0]) = y_0.

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        t_0     - left-hand endpoint of interval (initial condition is here)
        t_f     - right-hand endpoint of interval
        y_0    - initial x value: x0 = x(a)
        TOL   - maximum value of local truncation error estimate
        h_max  - maximum step size
        h_min  - minimum step size

    OUTPUT:
        t     - NumPy array of independent variable values
        y     - NumPy array of corresponding solution function values
    """

    if (method == "Fehlberg"):
        # Coefficients used to compute the independent variable argument of f

        c2  =   1/4
        c3  =   3/8
        c4  =   12/13
        c5  =   1
        c6  =   1/2

        # Coefficients used to compute the dependent variable argument of f

        a21 =   1/4
        a31 =   3/32
        a32 =   9/32
        a41 =   1932/2197
        a42 =   -7200/2197
        a43 =   7296/2197
        a51 =   439/216
        a52 =   -8
        a53 =   3680/513
        a54 =  -845/4104
        a61 =  -8/27
        a62 =   2
        a63 =  -3544/2565
        a64 =   1859/4104
        a65 =  -11/40

        # Coefficients used to compute 4th order RK estimate

        b1  =   25/216
        b3  =   1408/2565
        b4  =   2197/4104
        b5  =  -1/5
        b6 = 0
    
        b1_star  =   16/135
        b3_star  =   6656/12825
        b4_star  =   28561/56430
        b5_star  =  -9/50
        b6_star  =   2/55
        
    elif (method == "Cash-Karp"):
        # Coefficients used to compute the independent variable argument of f

        c2  =   1/5
        c3  =   3/10
        c4  =   3/5
        c5  =   1
        c6  =   7/8

        # Coefficients used to compute the dependent variable argument of f

        a21 =   1/5
        a31 =   3/40
        a32 =   9/40
        a41 =   3/10
        a42 =   -9/10
        a43 =   6/5
        a51 =   -11/54
        a52 =   5/2
        a53 =   -70/27
        a54 =   35/27
        a61 =   1631/55296
        a62 =   175/512
        a63 =   575/13824
        a64 =   44275/110592
        a65 =   253/4096

        # Coefficients used to compute 4th order RK estimate

        b1  =   2825/27648
        b3  =   18575/48384
        b4  =   13525/55296
        b5  =   277/14336
        b6  =   1/4
    
        b1_star  =   37/378
        b3_star  =   250/621
        b4_star  =   125/594
        b5_star  =   0
        b6_star  =   512/1771
        
    else:
        print("This method is not yet difined or incorrect!")
        
    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1  =   b1_star - b1
    r3  =   b3_star - b3
    r4  =   b4_star - b4
    r5  =   b5_star - b5
    r6  =   b6_star - b6

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
  
    t = t_0
    y = y_0
    h = h_max

    # Initialize arrays that will be returned

    T = np.array( [t] )
    Y = np.array( [y] )

    while t < t_f:

        # Adjust step size when we get to last interval

        if t + h > t_f:
            h = t_f - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.

        k1 = h * f(t, y)
        k2 = h * f(t + c2 * h, y + a21 * k1)
        k3 = h * f(t + c3 * h, y + a31 * k1 + a32 * k2)
        k4 = h * f(t + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)
        k5 = h * f(t + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k6 = h * f(t + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
    
        R = np.abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 ) / h
        if len( np.shape(R) ) > 0:
            r = np.max(R)
        if R <= TOL:
            t = t + h
            y = y + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
            T = np.append(T, t)
            Y = np.append(Y, [y], 0)

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( TOL / R )**0.25, 0.1 ), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            print( "Error: stepsize should be smaller than %e." % h_min )
            break

    # endwhile

    return (T, Y)

def Runge_Kutta_Dormand_Prince(f, t_0, t_f, y_0, h_min, h_max, TOL=1e-6):
    """Runge-Kutta-Dormand-Prince method to solve y' = f(x,t) with y(t[0]) = y_0.

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        t_0     - left-hand endpoint of interval (initial condition is here)
        t_f     - right-hand endpoint of interval
        y_0    - initial x value: x0 = x(a)
        TOL   - maximum value of local truncation error estimate
        h_max  - maximum step size
        h_min  - minimum step size

    OUTPUT:
        t     - NumPy array of independent variable values
        y     - NumPy array of corresponding solution function values
    """

    # Coefficients used to compute the independent variable argument of f

    c2  =   1/5
    c3  =   3/10
    c4  =   4/5
    c5  =   8/9
    c6  =   1
    c7  =   1

    # Coefficients used to compute the dependent variable argument of f

    a21 =   1/5
    a31 =   3/40
    a32 =   9/40
    a41 =   44/45
    a42 =   -56/15
    a43 =   32/9
    a51 =   19372/6561
    a52 =   -25360/2187
    a53 =   64448/6561
    a54 =  -212/729
    a61 =   9017/3168
    a62 =   -355/33
    a63 =   46732/5247
    a64 =   49/176
    a65 =  -5103/18656
    a71 =   35/384
    a72 =   0
    a73 =   500/1113
    a74 =   125/192
    a75 =  -2187/6784
    a76 =   11/84

    # Coefficients used to compute 4th order RK estimate

    b1  =   5179/57600
    b3  =   7571/16695
    b4  =   393/640
    b5  =  -92097/339200
    b6  =   187/2100
    b7  =   1/40
    
    b1_star  =   35/384
    b3_star  =   500/1113
    b4_star  =   125/192
    b5_star  =  -2187/6784
    b6_star  =   11/84
    b7_star  =   0.0
    
    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1  =   b1_star - b1
    r3  =   b3_star - b3
    r4  =   b4_star - b4
    r5  =   b5_star - b5
    r6  =   b6_star - b6
    r7  =   b7_star - b7

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
  
    t = t_0
    y = y_0
    h = h_max

    # Initialize arrays that will be returned

    T = np.array( [t] )
    Y = np.array( [y] )

    while t < t_f:

        # Adjust step size when we get to last interval

        if t + h > t_f:
            h = t_f - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.

        k1 = h * f(t, y)
        k2 = h * f(t + c2 * h, y + a21 * k1)
        k3 = h * f(t + c3 * h, y + a31 * k1 + a32 * k2)
        k4 = h * f(t + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)
        k5 = h * f(t + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k6 = h * f(t + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k7 = h * f(t + c7 * h, y + a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
    
        R = np.abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 + r7 * k7 ) / h
        if len( np.shape(R) ) > 0:
            r = np.max(R)
        if R <= TOL:
            t = t + h
            y = y + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7
            T = np.append(T, t)
            Y = np.append(Y, [y], 0)

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( TOL / R )**0.25, 0.1 ), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            print( "Error: stepsize should be smaller than %e." % h_min )
            break

    # endwhile

    return (T, Y)

def Runge_Kutta_Verner(f, t_0, t_f, y_0, h_min, h_max, TOL=1e-6):
    """Runge-Kutta-Verner method to solve y' = f(x,t) with y(t[0]) = y_0.

    INPUT:
        f     - function equal to dx/dt = f(x,t)
        t_0     - left-hand endpoint of interval (initial condition is here)
        t_f     - right-hand endpoint of interval
        y_0    - initial x value: x0 = x(a)
        TOL   - maximum value of local truncation error estimate
        h_max  - maximum step size
        h_min  - minimum step size

    OUTPUT:
        t     - NumPy array of independent variable values
        y     - NumPy array of corresponding solution function values
    """

    # Coefficients used to compute the independent variable argument of f

    c2  =   1/6
    c3  =   4/15
    c4  =   2/3
    c5  =   5/6
    c6  =   1
    c7  =   1/15
    c8  =   1

    # Coefficients used to compute the dependent variable argument of f

    a21 =   1/6
    a31 =   4/75
    a32 =   16/75
    a41 =   5/6
    a42 =   -8/3
    a43 =   5/2
    a51 =   -165/64
    a52 =   55/6
    a53 =   -425/64
    a54 =   85/96
    a61 =   12/5
    a62 =   -8
    a63 =   4015/612
    a64 =   -11/36
    a65 =   88/255
    a71 =   -8263/15000
    a72 =   124/75
    a73 =   -643/680
    a74 =   -81/250
    a75 =   2484/10625
    a76 =   0
    a81 =   3501/1720
    a82 =   -300/43
    a83 =   297275/52632
    a84 =   -319/2322
    a85 =   24068/84065
    a86 =   0
    a87 =   3850/26703

    # Coefficients used to compute 4th order RK estimate

    b1  =   13/160
    b3  =   2375/5984
    b4  =   5/16
    b5  =   12/85
    b6  =   3/44
    b7  =   0.0
    b8  =   0.0
    
    b1_star  =   3/40
    b3_star  =   875/2244
    b4_star  =   23/72
    b5_star  =   264/1955
    b6_star  =   0
    b7_star  =   125/11592
    b8_star  =   43/616
    
    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1  =   b1_star - b1
    r3  =   b3_star - b3
    r4  =   b4_star - b4
    r5  =   b5_star - b5
    r6  =   b6_star - b6
    r7  =   b7_star - b7
    r8  =   b8_star - b8

    # Set t and x according to initial condition and assume that h starts
    # with a value that is as large as possible.
  
    t = t_0
    y = y_0
    h = h_max

    # Initialize arrays that will be returned

    T = np.array( [t] )
    Y = np.array( [y] )

    while t < t_f:

        # Adjust step size when we get to last interval

        if t + h > t_f:
            h = t_f - t;

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.

        k1 = h * f(t, y)
        k2 = h * f(t + c2 * h, y + a21 * k1)
        k3 = h * f(t + c3 * h, y + a31 * k1 + a32 * k2)
        k4 = h * f(t + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)
        k5 = h * f(t + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k6 = h * f(t + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k7 = h * f(t + c7 * h, y + a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        k8 = h * f(t + c8 * h, y + a81 * k1 + a82 * k2 + a83 * k3 + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7)

        # Compute the estimate of the local truncation error.  If it's small
        # enough then we accept this step and save the 4th order estimate.
    
        R = np.abs( r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6 + r7 * k7 + r8 * k8) / h
        if len( np.shape(R) ) > 0:
            r = np.max(R)
        if R <= TOL:
            t = t + h
            y = y + b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7 + b8 * k8
            T = np.append(T, t)
            Y = np.append(Y, [y], 0)

        # Now compute next step size, and make sure that it is not too big or
        # too small.

        h = h * min( max( 0.84 * ( TOL / R )**0.25, 0.1 ), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            print( "Error: stepsize should be smaller than %e." % h_min )
            break

    # endwhile

    return (T, Y)

def Adams_Bashforth_Two_Step(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    t_, y_ = Original_Runge_Kutta_Method(f=f, t_0=t_0, t_f=t_f, y_0=y_0, h=h)
    y[0:2] = y_[0:2]
    for i in range(1, n-1):
        K1 = f(t[i],y[i])
        K2 = f(t[i-1],y[i-1])
        y[i+1] = y[i] + 0.5*h*(3*K1 - K2)
    return (t, y)

def Adams_Bashforth_Three_Step(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    t_, y_ = Original_Runge_Kutta_Method(f=f, t_0=t_0, t_f=t_f, y_0=y_0, h=h)
    y[0:3] = y_[0:3]
    for i in range(2, n-1):
        K1 = f(t[i],y[i])
        K2 = f(t[i-1],y[i-1])
        K3 = f(t[i-2],y[i-2])
        y[i+1] = y[i] + h*(23*K1-16*K2+5*K3)/12
    return (t, y)

def Adams_Bashforth_Four_Step(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    t_, y_ = Original_Runge_Kutta_Method(f=f, t_0=t_0, t_f=t_f, y_0=y_0, h=h)
    y[0:4] = y_[0:4]
    
    for i in range(3, n-1):
        K1 = f(t[i], y[i])
        K2 = f(t[i-1], y[i-1])
        K3 = f(t[i-2], y[i-2])
        K4 = f(t[i-3], y[i-3])
        y[i+1] = y[i] + h * (55 * K1 - 59 * K2 + 37 * K3 - 9 * K4) / 24
        
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Adams_Bashforth_Five_Step(f, t_0, t_f, y_0, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    t_, y_ = Original_Runge_Kutta_Method(f=f, t_0=t_0, t_f=t_f, y_0=y_0, h=h)
    y[0:5] = y_[0:5]
    for i in range(4, n-1):
        K1 = f(t[i],y[i])
        K2 = f(t[i-1],y[i-1])
        K3 = f(t[i-2],y[i-2])
        K4 = f(t[i-3],y[i-3])
        K5 = f(t[i-4],y[i-4])
        y[i+1] = y[i] + h*(1901*K1 - 2774*K2 + 2616*K3 - 1274*K4 + 251*K5)/720
    return (t, y)

def Adams_Predictor_Corrector(f, t_0, t_f, y_0, h):
    
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    y = np.zeros(n)  # Solution array
    y[0] = y_0       # intial condition 

    
    # using RK4 to obtain the first 3 points
    for i in range(0, n-1):
        if i in range(0,3):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + (h/2.0), y[i] +(k1/2.0))
            k3 = h * f(t[i] + (h/2.0), y[i] + (k2/2.0))
            k4 = h * f(t[i] + h, y[i] + k3)
        
            y[i+1] = y[i] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
            
        else:
            ##  explicit Adams-Bashforth method as predictor:
            y[i+1] = y[i] + h*(55.0 * f(t[i],y[i]) - 59.0 * f(t[i-1],y[i-1]) + 37.0 * f(t[i-2],y[i-2]) - 9.0 * f(t[i-3],y[i-3]))/24.0
             
            ## three-step implicit Adams-Moulton method as a corrector:
            y[i+1] = y[i] + h*(9.0 * f(t[i+1], y[i + 1]) + 19.0 * f(t[i],y[i]) - 5.0 * f(t[i-1],y[i-1]) + f(t[i-2],y[i-2]))/24.0
             
             
    df = pd.DataFrame({'t': t, 'y': y}, columns=['t', 'y'])
    
    return df

def Runge_Kutta_Method_for_Systems(f, t_0, t_f, y_init, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    m = len(y_init)
    y = np.full(shape=(n, m), fill_value=np.nan, dtype=np.float64)   # Solution array
    y[0, :] = y_init  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1, :])
        k_2 = f(t[i-1] + 0.5*h, y[i-1, :] + 0.5*h*k_1)
        k_3 = f(t[i-1] + 0.5*h, y[i-1, :] + 0.5*h*k_2)
        k_4 = f(t[i-1] + h, y[i-1, :] + h*k_3)
        y[i, :] = y[i-1, :] + h*((1/6)*k_1 + (1/3)*k_2 + (1/3)*k_3 + (1/6)*k_4)
        
    df_t = pd.DataFrame(data=t, columns=["t"])
    df_y = pd.DataFrame(data=y, columns=[f"y{i+1}" for i in range(m)])
    df = pd.concat(objs=[df_t, df_y], axis="columns")
        
    return df

### TP 6
def Runge_Kutta_Method_for_System_TP6(f, t_0, t_f, y_init, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    m = len(y_init)
    y = np.full(shape=(n, m), fill_value=np.nan, dtype=np.float64)   # Solution array
    y[0, :] = y_init  # Initial condition

    for i in range(1, n):
        k_1 = f(t[i-1], y[i-1, :])
        k_2 = f(t[i-1] + 0.5*h, y[i-1, :] + 0.5*h*k_1)
        k_3 = f(t[i-1] + 0.5*h, y[i-1, :] + 0.5*h*k_2)
        k_4 = f(t[i-1] + h, y[i-1, :] + h*k_3)
        y[i, :] = y[i-1, :] + h*((1/6)*k_1 + (1/3)*k_2 + (1/3)*k_3 + (1/6)*k_4)
        
    return y

def Linear_Shooting_Method(f1, f2, t_0, t_f, alpha, beta, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)
    
    # Apply Runge-Kutta for systems 
    y_init_1 = np.array([alpha, 0], dtype=np.float64)
    y_init_2 = np.array([0, 1], dtype=np.float64)
    u = Runge_Kutta_Method_for_System(f1, t_0, t_f, y_init_1, h)
    v = Runge_Kutta_Method_for_System(f2, t_0, t_f, y_init_2, h)
    
    # w is a vector solution for y(t) and y'(t), respectively
    w = np.zeros([2,n])
    w[0,0] = alpha
    w[1,0] = (beta - u[n-1,0])/v[n-1,0]
    
    for i in range(1,n):
        w[0,i] = u[i,0] + w[1,0]*v[i,0]
        w[1,i] = u[i,1] + w[1,0]*v[i,1]

    return (t, w)

def Nonlinear_Shooting_Algorithm(f, f_y, f_y_p, t_0, t_f, alpha, beta, h, M=1000, TOL=1e-6):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t)

    w1 = np.zeros(n)  
    w2 = np.zeros(n)
    k = 1
    a = t_0
    b = t_f
    TK = (beta - alpha)/(b - a)

    print("  x" "     " "W1""     " "W2")
    while k <= M:

        w1[0] = alpha
        w2[0] = TK
        u1    = 0
        u2    = 1

        for i in range(1, n):
            x = a + (i-1)*h     #step 5

            t = x + 0.5*h
            k11 = h*w2[i-1]     #step 6
            k12 = h*f(x, w1[i-1], w2[i-1])
            k21 = h*(w2[i-1] + (1/2)*k12)
            k22 = h*f(t, w1[i-1] + (1/2)*k11, w2[i-1] + (1/2)*k12)
            k31 = h*(w2[i-1] + (1/2)*k22)
            k32 = h*f(t, w1[i-1] + (1/2)*k21, w2[i-1] + (1/2)*k22)
            
            t   = x + h
            k41 = h*(w2[i-1]+k32)
            k42 = h*f(t, w1[i-1] + k31, w2[i-1] + k32)
            w1[i] = w1[i-1] + (k11 + 2*k21 + 2*k31 + k41)/6
            w2[i] = w2[i-1] + (k12 + 2*k22 + 2*k32 + k42)/6   
            kp11 = h*u2
            kp12 = h*(f_y(x,w1[i-1],w2[i-1])*u1 + f_y_p(x,w1[i-1], w2[i-1])*u2)
            
            t    = x + 0.5*(h)
            kp21 = h*(u2 + (1/2)*kp12)
            kp22 = h*((f_y(t, w1[i-1], w2[i-1])*(u1 + (1/2)*kp11)) + f_y_p(x+h/2, w1[i-1],w2[i-1])*(u2 +(1/2)*kp12))
            kp31 = h*(u2 + (1/2)*kp22)
            kp32 = h*((f_y(t, w1[i-1], w2[i-1])*(u1 + (1/2)*kp21)) + f_y_p(x+h/2, w1[i-1],w2[i-1])*(u2 +(1/2)*kp22))
            
            t    = x + h
            kp41 = h*(u2 + kp32)
            kp42 = h*(f_y(t, w1[i-1], w2[i-1])*(u1+kp31) + f_y_p(x + h, w1[i-1], w2[i-1])*(u2 + kp32))
            u1 = u1 + (1/6)*(kp11 + 2*kp21 + 2*kp31 + kp41)
            u2 = u2 + (1/6)*(kp12 + 2*kp22 + 2*kp32 + kp42)


        r = abs(w1[n-1] - beta)
        if r <= TOL:
            for i in range(n):
                x = a + i*h
                print("%.2f %.4f %.4f" %(x, w1[i], w2[i]))
            return

        TK = TK - (w1[n-1]-beta)/u1

        k = k+1


    print("Maximum number of iterations exceeded!")   
    return 0

def Linear_Finite_Difference_Method(p, q, r, t_0, t_f, alpha, beta, h):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t) - 1
    
    a = np.zeros([n+1])  # 2 cuz we need w_0 and w_n+1
    b = np.zeros([n+1])  
    c = np.zeros([n+1])  
    d = np.zeros([n+1])
    
    x = t_0 + h
    a[1] = 2.0 + (h**2)*q(x)
    b[1] = -1.0 + (h/2)*p(x)
    d[1] = -(h**2)*r(x) + (1 + (h/2)*p(x))*alpha
    
    for i in range(2,n):
        x = t_0 + i*h
        a[i] = 2.0 + (h**2)*q(x)
        b[i] = -1.0 + (h/2)*p(x)
        c[i] = -1.0 - (h/2)*p(x)
        d[i] = -(h**2)*r(x)
        
    x = t_f - h
    a[n] = 2.0 + (h**2)*q(x)
    c[n] = -1.0 - (h/2)*p(x)
    d[n] = -(h**2)*r(x) + (1.0 - (h/2)*p(x))*beta
    
    l = np.zeros([n+1])
    u = np.zeros([n+1])
    z = np.zeros([n+1])
    
    # Crout algorithm
    l[1] = a[1]
    u[1] = b[1]/a[1]
    z[1] = d[1]/l[1]
    
    for i in range(2,n):
        l[i] = a[i]-c[i]*u[i-1]
        u[i] = b[i]/l[i]
        z[i] = (d[i] - c[i]*z[i-1])/l[i]
        
    l[n] = a[n] - c[n]*u[n-1]
    z[n] = (d[n] -c[n]*z[n-1])/l[n]
    
    w = np.zeros([n+1])
    
    w[0] = alpha
    w[n] = beta
    w[n-1] = z[n]
    
    for i in range(n-1,0,-1):
        w[i] = z[i] - u[i]*w[i+1]

    return (t, w)

def Newton_Method_for_Systems(f, jacobian_f, x_init, M=1000, TOL=1e-6):
    x_last = x_init

    for k in range(1, M):
        # Solve J(xn)*( xn+1 - xn ) = -F(xn):
        J = jacobian_f(x_last)
        F = f(x_last)

        diff = np.linalg.solve(J, -F)
        x_last = x_last + diff

        # Stop condition:
        if np.linalg.norm(diff) < TOL:
            print('Convergence!, with number of iterations:', k )
            break

    return x_last

def Nonlinear_Finite_Difference_Algorithm(f, dfy, dfyp, t_0, t_f, alpha, beta, h, M=1000, TOL=1e-6):
    t = np.arange(t_0, t_f + h, h)  # Time points
    n = len(t) - 2
    print(t)

    w = np.zeros([n+2])
    a = np.zeros([n+1])
    b = np.zeros([n+1])
    c = np.zeros([n+1])
    d = np.zeros([n+1])

    w[0] = alpha
    w[n+1] = beta

    for i in range(1, n+1):
        w[i] = alpha + i*(beta-alpha)*h/(t_f - t_0)

    k = 1

    while(k <= M):
        
        x = t_0 + h
        t = (w[2] - alpha)/(2*h)
        a[1] = 2 + h**2*dfy(x,w[1],t)
        b[1] = -1 + (h/2)*dfyp(x,w[1],t)
        d[1] = -(2*w[1] - w[2] - alpha + h**2*f(x,w[1],t))

        for i in range(2,n):
            x = t_0 + i*h
            t = (w[i+1] - w[i-1])/(2*h)
            a[i] = 2 + h**2*dfy(x,w[i],t)
            b[i] = -1 + (h/2)*dfyp(x,w[i],t)
            c[i] = -1 - (h/2)*dfyp(x,w[i],t)
            d[i] = -(2*w[i] - w[i+1] - w[i-1] + h**2*f(x,w[i],t))

        x = t_f - h
        t = (beta - w[n-1])/(2*h)
        a[n] = 2 + h**2*dfy(x,w[n],t)
        c[n] = -1 - (h/2)*dfyp(x,w[n],t)
        d[n] = -(2*w[n] - beta - w[n-1] + h**2*f(x,w[n],t))

        # Solve tridiagonal system for Crout algorithm.
        
        l = np.zeros([n+1])
        u = np.zeros([n+1])
        z = np.zeros([n+1])

        l[1] = a[1]
        u[1] = b[1]/a[1]
        z[1] = d[1]/l[1]

        for i in range(2,n):
            l[i] = a[i] - c[i]*u[i-1]
            u[i] = b[i]/l[i]
            z[i] = (d[i] - c[i]*z[i-1])/l[i]

        l[n] = a[n] - c[n]*u[n-1]
        z[n] = (d[n] - c[n]*z[n-1])/l[n]

        v = np.zeros([n+2])

        v[n] = z[n]
        w[n] = w[n] + v[n]

        for i in range(n-1,0,-1):
            v[i] = z[i] - u[i]*v[i+1]
            w[i] = w[i] + v[i]

        if(np.linalg.norm(v) <= TOL):
            return w

        k = k+1

    print("Max of iterations, performance over without exit")
    return 

# Runge Kutta 40
def RungeKutta40(f: Callable[[np.float64, np.float64], np.float64],
                t_span: np.ndarray,
                y_init: np.float64,
                n: np.int64) -> pd.DataFrame:
    h = (t_span[1] - t_span[0]) / n
    t = np.linspace(start=t_span[1], stop=t_span[1], num=n+1, dtype=np.float64)
    y = np.full_like(a=t, fill_value=np.nan, dtype=np.float64)
    y[0] = y_init
    for i in range(0, n, 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    df = pd.DataFrame(data={"t": t, "y": y}, dtype=np.float64)
    return df

# Runge Kutta 41
def RungeKutta41(f: Callable[[np.float64, np.float64], np.float64],
                t_span: np.ndarray,
                y_init: np.float64,
                n: np.int64) -> pd.DataFrame:
    h = (t_span[1] - t_span[0]) / n
    t = np.linspace(start=t_span[0], stop=t_span[1], num=n+1, dtype=np.float64)
    y = np.full_like(a=t, fill_value=np.nan, dtype=np.float64)
    y[0] = y_init
    # Butcher Table
    c = np.array(object=[0, 1/2, 1/2, 1], 
                dtype=np.float64)
    a = np.array(object=[[0, 0, 0, 0],
                        [1/2, 0, 0, 0],
                        [0, 1/2, 0, 0],
                        [0, 0, 1, 0]], 
                dtype=np.float64)
    b = np.array(object=[1/6, 1/3, 1/3, 1/6],
                dtype=np.float64)
    for i in range(0, n, 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + c[1] * h, y[i] + (a[1, 0] * k1) * h)
        k3 = f(t[i] + c[2] * h, y[i] + (a[2, 0] * k1 + a[2, 1] * k2) * h)
        k4 = f(t[i] + c[2] * h, y[i] + (a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3) * h)
        y[i+1] = y[i] + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4)
    df = pd.DataFrame(data={"t": t, "y": y}, dtype=np.float64)
    return df

# Runge Kutta Fahlberg 45
def RungeKuttaFehlberg45(
    f: Callable[[np.float64, np.float64], np.float64],
    t_span: np.ndarray,
    y_init: np.float64,
    h_max: Optional[np.float64] = 2.5e-1,
    h_min: Optional[np.float64] = 1.e-4,
    tol: Optional[np.float64] = 1.e-10
) -> pd.DataFrame:
    t = t_span[0]
    y = y_init
    h = h_max
    i = 0
    df = pd.DataFrame(data={"t": [t], "h": [None], "y": [y]})
    # # TODO: Butcher tableau of Runge-Kutta-Fehlberg of order 4 and 5
    c = np.array(object=[0., 1./4., 3./8., 12./13., 1., 1./2.], dtype=np.float64)
    a = np.array(object=[[0., 0., 0., 0., 0.],
                        [1./4., 0., 0., 0., 0.],
                        [3./32, 9./32., 0., 0., 0.],
                        [1932./2197., -7200./2197., 7296./2197., 0., 0.],
                        [439./216., -8., 3680./513., -845./4104., 0.],
                        [-8./27., 2., -3544./2565., 1859./4104., -11./40.]], dtype=np.float64)
    b = np.array(object=[25./216., 0., 1408./2565., 2197./4104., -1./5.], dtype=np.float64)
    db = np.array(object=[1./360., 0., -128./4275., -2197./75240., 1./50., 2./55.], dtype=np.float64)
    flag = 1
    while flag:
        k0 = f(t, y)
        k1 = f(t + c[1]*h, y + (a[1, 0] * k0) * h)
        k2 = f(t + c[2]*h, y + (a[2, 0] * k0 + a[2, 1] * k1) * h)
        k3 = f(t + c[3]*h, y + (a[3, 0] * k0 + a[3, 1] * k1 + a[3, 2] * k2) * h)
        k4 = f(t + c[4]*h, y + (a[4, 0] * k0 + a[4, 1] * k1 + a[4, 2] * k2 + a[4, 3] * k3) * h)
        k5 = f(t + c[5]*h, y + (a[5, 0] * k0 + a[5, 1] * k1 + a[5, 2] * k2 + a[5, 3] * k3 + a[5, 4] * k4) * h)
        r = abs(db[0]*k0 + db[1]*k1 + db[2]*k2 + db[3]*k3 + db[4]*k4 + db[5]*k5)
        # TODO: Check if the truncation error r is acceptable
        if r <= tol:
            t = t + h
            y = y + (b[0]*k0 + b[1]*k1 + b[2]*k2 + b[3]*k3 + b[4]*k4) * h
            i = i + 1
            df.loc[i, :] = [t, h, y]
        d = 0.84*(tol / r)**(1./4.)
        # d = ((tol*h/(2*r))**0.25)
        # TODO: Control the factor d by 0.1 <= h_new / h_old <= 4.0
        if d < 0.1:
            d = 0.1
        elif d > 4.0:
            d = 4.0
        h = d * h
        # TODO: Control h with constraint h <= h_max
        if h > h_max:
            h = h_max
        # TODO: Check if last step reached
        if t >= t_span[1]:
            flag = 0
            print(f":)Successfully completed with {i} steps!")
        elif t + h > t_span[1]:
            h = t_span[1] - t
        # TODO: Check if h is too small
        elif h < h_min:
            flag = 0
            print(f":( At i = {i}, h_i = {h} < h_min = {h_min}")
    return df

# Runge Kutta Dor
def RungeKuttaFehlberg45(
    f: Callable[[np.float64, np.float64], np.float64],
    t_span: np.ndarray,
    y_init: np.float64,
    h_max: Optional[np.float64] = 2.5e-1,
    h_min: Optional[np.float64] = 1.e-4,
    tol: Optional[np.float64] = 1.e-10
) -> pd.DataFrame:
    t = t_span[0]
    y = y_init
    h = h_max
    i = 0
    df = pd.DataFrame(data={"t": [t], "h": [None], "y": [y]})
    # # TODO: Butcher tableau of Runge-Kutta-Fehlberg of order 4 and 5
    c = np.array(object=[0., 1./4., 3./8., 12./13., 1., 1./2.], dtype=np.float64)
    a = np.array(object=[[0., 0., 0., 0., 0.],
                        [1./4., 0., 0., 0., 0.],
                        [3./32, 9./32., 0., 0., 0.],
                        [1932./2197., -7200./2197., 7296./2197., 0., 0.],
                        [439./216., -8., 3680./513., -845./4104., 0.],
                        [-8./27., 2., -3544./2565., 1859./4104., -11./40.]], dtype=np.float64)
    b = np.array(object=[25./216., 0., 1408./2565., 2197./4104., -1./5.], dtype=np.float64)
    db = np.array(object=[1./360., 0., -128./4275., -2197./75240., 1./50., 2./55.], dtype=np.float64)
    flag = 1
    while flag:
        k0 = f(t, y)
        k1 = f(t + c[1]*h, y + (a[1, 0] * k0) * h)
        k2 = f(t + c[2]*h, y + (a[2, 0] * k0 + a[2, 1] * k1) * h)
        k3 = f(t + c[3]*h, y + (a[3, 0] * k0 + a[3, 1] * k1 + a[3, 2] * k2) * h)
        k4 = f(t + c[4]*h, y + (a[4, 0] * k0 + a[4, 1] * k1 + a[4, 2] * k2 + a[4, 3] * k3) * h)
        k5 = f(t + c[5]*h, y + (a[5, 0] * k0 + a[5, 1] * k1 + a[5, 2] * k2 + a[5, 3] * k3 + a[5, 4] * k4) * h)
        r = abs(db[0]*k0 + db[1]*k1 + db[2]*k2 + db[3]*k3 + db[4]*k4 + db[5]*k5)
        # TODO: Check if the truncation error r is acceptable
        if r <= tol:
            t = t + h
            y = y + (b[0]*k0 + b[1]*k1 + b[2]*k2 + b[3]*k3 + b[4]*k4) * h
            i = i + 1
            df.loc[i, :] = [t, h, y]
        d = 0.84*(tol / r)**(1./4.)
        # d = ((tol*h/(2*r))**0.25)
        # TODO: Control the factor d by 0.1 <= h_new / h_old <= 4.0
        if d < 0.1:
            d = 0.1
        elif d > 4.0:
            d = 4.0
        h = d * h
        # TODO: Control h with constraint h <= h_max
        if h > h_max:
            h = h_max
        # TODO: Check if last step reached
        if t >= t_span[1]:
            flag = 0
            print(f":)Successfully completed with {i} steps!")
        elif t + h > t_span[1]:
            h = t_span[1] - t
        # TODO: Check if h is too small
        elif h < h_min:
            flag = 0
            print(f":( At i = {i}, h_i = {h} < h_min = {h_min}")
    return df

##### Runge Kutta Method Order 4 - Solve for IVPs with 2 equations of First derivative given.
def runge_kutta_4_2eqs(f, g, t_span, y1_0, y2_0, num_steps):
    '''
    Note: `f` assigned as `y1` and `g` assigned as `y2`
    
    # Define the initial conditions and parameters
    t_span = (0, 1)  # Start and end points for t
    y1_0 = 2  # Initial value of y1
    y2_0 = -3  # Initial value of y2
    num_steps = 20  # Number of iterations

    # Call the Runge-Kutta solver
    runge_kutta_4_2eqs(f, g, t_span, y1_0, y2_0, num_steps)
    '''
    t0, tf = t_span
    h = (tf - t0) / num_steps

    t = t0
    y1 = y1_0
    y2 = y2_0
    
    t_list, y1_list, y2_list = [], [], []
    
    y1_list.append(y1)
    t_list.append(t)
    y2_list.append(y2)

    for _ in range(num_steps):
        # Compute the four intermediate steps
        k1_y1 = h * f(t, y1, y2)
        k1_y2 = h * g(t, y1, y2)

        k2_y1 = h * f(t + h/2, y1 + k1_y1/2, y2 + k1_y2/2)
        k2_y2 = h * g(t + h/2, y1 + k1_y1/2, y2 + k1_y2/2)

        k3_y1 = h * f(t + h/2, y1 + k2_y1/2, y2 + k2_y2/2)
        k3_y2 = h * g(t + h/2, y1 + k2_y1/2, y2 + k2_y2/2)

        k4_y1 = h * f(t + h, y1 + k3_y1, y2 + k3_y2)
        k4_y2 = h * g(t + h, y1 + k3_y1, y2 + k3_y2)

        # Update the values of y1 and y2 using the weighted average of the four steps
        y1 += (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6
        y2 += (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6

        # Update the value of t
        t += h
        
        y1_list.append(y1)
        t_list.append(t)
        y2_list.append(y2)

        # Print the result or store it in arrays if needed
    df = pd.DataFrame({'t': t_list, 'y1': y1_list, 'y2': y2_list})
                      
    return df 

##### Solve for IVP with 2nd Derivative given.
def runge_kutta_4_system_2nd_derivative(f, x0, y0, y_prime0, x_end, N):
    '''
    # Define function
    def f(x, y, y_prime):
        return 4*y_prime - 5*y - np.cos(x)

    # Initial values
    x0 = 0
    y0 = 0
    y_prime0 = 1

    # End value of x
    x_end = 1

    # Number of steps
    N = 20

    # Solve the IVP using the fourth-order Runge-Kutta method
    runge_kutta_4_system_2nd_derivative(f, x0, y0, y_prime0, x_end, N)
    '''
    h = (x_end - x0) / N
    x = x0
    y = y0
    y_prime = y_prime0
    x_list, y_list, y_prime_list = [], [], []
    
    y_list.append(y)
    x_list.append(x)
    y_prime_list.append(y_prime)
    
    for i in range(N):
        k1_y = h * y_prime
        k1_z = h * f(x, y, y_prime)

        k2_y = h * (y_prime + k1_z/2)
        k2_z = h * f(x + h/2, y + k1_y/2, y_prime + k1_z/2)

        k3_y = h * (y_prime + k2_z/2)
        k3_z = h * f(x + h/2, y + k2_y/2, y_prime + k2_z/2)

        k4_y = h * (y_prime + k3_z)
        k4_z = h * f(x + h, y + k3_y, y_prime + k3_z)

        y = y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        y_prime = y_prime + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        x = x + h
        y_list.append(y)
        x_list.append(x)
        y_prime_list.append(y_prime)
        
    df = pd.DataFrame({'x': x_list, 'y': y_list, 'y_prime': y_prime_list}, columns=['x', 'y', 'y_prime'])

    return df