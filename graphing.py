import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.widgets import Slider

# --------------------------------------------------
# SCALAR BROADCAST HELPER
# --------------------------------------------------
def broadcast_if_scalar(result, xvals):
    """
    If 'result' is a scalar (int, float) or a 0D NumPy array,
    broadcast it to match the shape of 'xvals'.
    """
    if np.isscalar(result) or (isinstance(result, np.ndarray) and result.shape == ()):
        return np.full_like(xvals, result)
    return result

# --------------------------------------------------
# CRITICAL POINTS
# --------------------------------------------------
def find_critical_points(f_expr):
    """
    Solve f'(x) = 0 to find critical points for auto-scaling.
    Only keep real solutions.
    """
    x_symbol = sp.Symbol('x')
    f_prime_expr = sp.diff(f_expr, x_symbol)  
    critical_points = sp.solve(f_prime_expr, x_symbol)  
    return [float(point) for point in critical_points if point.is_real]

# --------------------------------------------------
# TANGENT LINE
# --------------------------------------------------
def compute_tangent_line(f_expr, a):
    """
    Return a vectorized function for the tangent line of f_expr at x=a.
    Tangent line: y = f'(a)*(x - a) + f(a)
    """
    x_symbol = sp.Symbol('x')
    f_prime_expr = sp.diff(f_expr, x_symbol)
    f_lambdified = sp.lambdify(x_symbol, f_expr, 'numpy')
    f_prime_lambdified = sp.lambdify(x_symbol, f_prime_expr, 'numpy')

    f_a = f_lambdified(a)
    f_prime_a = f_prime_lambdified(a)

    def tangent_function(x_vals):
        # Return an array for all x_vals
        y_vals = f_prime_a * (x_vals - a) + f_a
        return broadcast_if_scalar(y_vals, x_vals)

    return tangent_function

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
def plot_function_and_derivatives(f_expr, a, num_points=400):
    """
    Plots:
    1) f(x) in blue
    2) f'(x) in red (dashed)
    3) f''(x) in green (dotted)
    4) Tangent line at x=a in orange (dashdot)
    Allows interactive zooming with sliders for X Min and X Max.
    """

    x_symbol = sp.Symbol('x')

    # Convert to NumPy-friendly lambdas
    f_lambdified = sp.lambdify(x_symbol, f_expr, 'numpy')
    f_prime_expr = sp.diff(f_expr, x_symbol)
    f_prime_lambdified = sp.lambdify(x_symbol, f_prime_expr, 'numpy')
    f_double_prime_expr = sp.diff(f_prime_expr, x_symbol)
    f_double_prime_lambdified = sp.lambdify(x_symbol, f_double_prime_expr, 'numpy')

    # Tangent function
    tangent_lambdified = compute_tangent_line(f_expr, a)

    # Auto-range based on critical points
    critical_points = find_critical_points(f_expr)
    if critical_points:
        x_min, x_max = min(critical_points) - 5, max(critical_points) + 5
    else:
        x_min, x_max = -10, 10

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)  # space for sliders

    def update(val):
        ax.clear()
        x_range = (slider_xmin.val, slider_xmax.val)
        x = np.linspace(x_range[0], x_range[1], num_points)

        # Evaluate original function, derivatives, tangent
        y = f_lambdified(x)
        y_prime = f_prime_lambdified(x)
        y_double_prime = f_double_prime_lambdified(x)
        y_tangent = tangent_lambdified(x)

        # Broadcast any scalars (in case user typed something constant)
        y = broadcast_if_scalar(y, x)
        y_prime = broadcast_if_scalar(y_prime, x)
        y_double_prime = broadcast_if_scalar(y_double_prime, x)
        y_tangent = broadcast_if_scalar(y_tangent, x)

        # Plot everything
        # Disable scientific notation and offset so the point shows up exactly where we expect
        ax.ticklabel_format(useOffset=False, style='plain', axis='both')
        ax.plot(x, y, label=f"f(x) = {f_expr}", color='blue')
        ax.plot(x, y_prime, label=f"f'(x) = {f_prime_expr}", color='red', linestyle="dashed")
        ax.plot(x, y_double_prime, label=f"f''(x) = {f_double_prime_expr}", color='green', linestyle="dotted")
        ax.plot(x, y_tangent, label=f"Tangent at x={a}", color='orange', linestyle="dashdot")

        # Mark tangent point
        f_val_at_a = f_lambdified(a)
        ax.scatter([a], [f_val_at_a], color='black', zorder=3, label="Tangent Point")

        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()
        ax.set_title("Function, Derivatives, and Tangent Line")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        fig.canvas.draw_idle()

    # Sliders
    slider_xmin_ax = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_xmax_ax = plt.axes([0.15, 0.05, 0.65, 0.03])
    slider_xmin = Slider(slider_xmin_ax, 'X Min', -50, 50, valinit=x_min)
    slider_xmax = Slider(slider_xmax_ax, 'X Max', -50, 50, valinit=x_max)
    slider_xmin.on_changed(update)
    slider_xmax.on_changed(update)

    update(None)  # Draw initial plot
    plt.show()

# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------
def main():
    """
    Main loop: asks user for a function and a tangent point, then prints the
    derivative equation, the value of the derivative at x=a, and f(a),
    then plots them.
    Type (yes/no) when prompted if you want to enter another function.
    """
    while True:
        try:
            user_input = input("\nEnter a mathematical function in terms of x (e.g., x**2 - 2*x + 1): ")
            f_expr = sp.sympify(user_input)
            a = float(input("Enter the x-value where you want the tangent line: "))
            # Compute the symbolic derivative
            derivative_symbolic = sp.diff(f_expr, sp.Symbol('x'))
            # Print derivative equation
            print(f"\nSymbolic derivative: {derivative_symbolic}")

            # Evaluate derivative at x=a
            f_prime_lambdified = sp.lambdify(sp.Symbol('x'), derivative_symbolic, 'numpy')
            derivative_at_a = f_prime_lambdified(a)
            print(f"f'({a}) = {derivative_at_a}")

            # Evaluate f(a)
            f_lambdified = sp.lambdify(sp.Symbol('x'), f_expr, 'numpy')
            f_val_at_a = f_lambdified(a)
            print(f"f({a}) = {f_val_at_a}")

            # Now plot
            plot_function_and_derivatives(f_expr, a)
        except Exception as e:
            print(f"Error: {e}")

        cont = input("\nDo you want to enter another function? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting program. Goodbye!")
            break

# --------------------------------------------------
# OPTIONAL TEST CASES
# --------------------------------------------------
def run_tests():
    """
    Test the plotting function with some sample functions and tangent points.
    Use these if you want a quick demonstration without manual input.
    """
    test_cases = [
        # function, tangent point
        ("x**2 - 2*x + 1", 3),  # Quadratic
        ("sin(x) + 2", 0),      # Sine shifted up
        ("5", 2),               # Constant
        ("2*x - 4", 0)          # Linear
    ]
    for func, point in test_cases:
        print(f"\nTest case: f(x) = {func}, tangent at x={point}")
        expr = sp.sympify(func)
        plot_function_and_derivatives(expr, point)

if __name__ == "__main__":
    # COMMENT OUT THIS LINE IF YOU WANT TO RUN THE TESTS:
    main()

    # UNCOMMENT THIS LINE TO RUN PRE-DEFINED TESTS:
    # run_tests()