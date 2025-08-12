### The Complete Mathematical Formulation

Let's define our components rigorously.

**1. Inputs:**
*   A query vector `q` from the input sequence at **absolute position `m`**.
*   A key vector `k` from the input sequence at **absolute position `n`**.
*   The embedding dimension is `d` (e.g., `d=64`). This must be an even number.

**2. The Positional Frequency (`θ_i`):**
First, we define a set of `d/2` distinct angular frequencies. These are constant and predetermined. The `i`-th frequency `θ_i` is given by the formula:
$$
\theta_i = \text{base}^{-\frac{2i}{d}} \quad \text{for } i \in \{0, 1, \dots, \frac{d}{2}-1\}
$$
Typically, `base = 10000`. This creates frequencies that range from very high (`θ_0 = 1.0`) to very low (e.g., `θ_{d/2 - 1} = 10000^{-1} = 0.0001`).

**3. The Rotation Function `f(x, p)`:**
This is the core function. It takes a vector `x` (which could be `q` or `k`) and its position `p` (which could be `m` or `n`) and applies the rotation.

We treat the `d`-dimensional vector `x = (x_0, x_1, x_2, \dots, x_{d-1})` as a sequence of `d/2` pairs:
$$
(x_0, x_1), (x_2, x_3), \dots, (x_{d-2}, x_{d-1})
$$
The `i`-th pair is `(x_{2i}, x_{2i+1})`.

The function `f` rotates the `i`-th pair of the vector `x` by an angle of `p \cdot \theta_i`. Notice how the **position `p` is a multiplier here!**

Using a 2D rotation matrix for the `i`-th pair:
$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(p \cdot \theta_i) & -\sin(p \cdot \theta_i) \\ \sin(p \cdot \theta_i) & \cos(p \cdot \theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$
This gives us the rotated components:
$$
\begin{align*}
x'_{2i} &= x_{2i} \cos(p \theta_i) - x_{2i+1} \sin(p \theta_i) \\
x'_{2i+1} &= x_{2i} \sin(p \theta_i) + x_{2i+1} \cos(p \theta_i)
\end{align*}
$$

**4. Applying the Rotation to Query and Key:**
Now we apply this function `f` to our query `q` at position `m` and our key `k` at position `n`.

The transformed query `q'_m` has components:
$$
\begin{align*}
q'_{m, 2i} &= q_{2i} \cos(m \theta_i) - q_{2i+1} \sin(m \theta_i) \\
q'_{m, 2i+1} &= q_{2i} \sin(m \theta_i) + q_{2i+1} \cos(m \theta_i)
\end{align*}
$$
The transformed key `k'_n` has components:
$$
\begin{align*}
k'_{n, 2i} &= k_{2i} \cos(n \theta_i) - k_{2i+1} \sin(n \theta_i) \\
k'_{n, 2i+1} &= k_{2i} \sin(n \theta_i) + k_{2i+1} \cos(n \theta_i)
\end{align*}
$$

**5. The Final Attention Score (The Payoff):**
The attention score is the dot product `q'_m \cdot k'_n`.
$$
\text{AttentionScore} = \sum_{j=0}^{d-1} q'_{m,j} k'_{n,j}
$$
Let's expand this sum by looking at the contribution from each `i`-th pair:
$$
\text{PairContribution}_i = q'_{m, 2i} k'_{n, 2i} + q'_{m, 2i+1} k'_{n, 2i+1}
$$
If we substitute the expressions from step 4 into this and perform a bit of algebraic expansion, it gets messy. However, there's a more elegant way using complex numbers.

### The Elegant View with Complex Numbers

Let's represent the `i`-th pair `(x_{2i}, x_{2i+1})` as a complex number `z_i = x_{2i} + j \cdot x_{2i+1}` (where `j` is the imaginary unit).

A rotation by an angle `\alpha` in the complex plane is simply multiplication by `e^{j\alpha}` (since `e^{j\alpha} = \cos\alpha + j\sin\alpha`).

So, the rotation function `f(x, p)` can be written as:
For each `i` from `0` to `d/2 - 1`:
$$
z'_{i} = (x_{2i} + j \cdot x_{2i+1}) \cdot e^{j \cdot p \cdot \theta_i}
$$

Now let's apply this to `q` and `k`. Let the complex representations be `q_c` and `k_c`.
$$
q'_{m, c, i} = (q_{2i} + j \cdot q_{2i+1}) \cdot e^{j m \theta_i}
$$
$$
k'_{n, c, i} = (k_{2i} + j \cdot k_{2i+1}) \cdot e^{j n \theta_i}
$$

The dot product of two vectors is the sum of the real parts of the multiplication of one vector's complex representation with the conjugate of the other (`Re(z_1 z_2^*)`).
So the contribution from pair `i` to the dot product is:
$$
\begin{align*}
\text{PairContribution}_i &= \text{Re} \left( q'_{m, c, i} \cdot (k'_{n, c, i})^* \right) \\
&= \text{Re} \left( (q_{c, i} e^{j m \theta_i}) \cdot (k_{c, i} e^{j n \theta_i})^* \right) \\
&= \text{Re} \left( q_{c, i} e^{j m \theta_i} \cdot k_{c, i}^* e^{-j n \theta_i} \right) \\
&= \text{Re} \left( (q_{c, i} k_{c, i}^*) \cdot e^{j(m-n)\theta_i} \right)
\end{align*}
$$
Let's break down that final term `(q_{c, i} k_{c, i}^*)`:
$$
q_{c, i} k_{c, i}^* = (q_{2i} + j q_{2i+1})(k_{2i} - j k_{2i+1}) = (q_{2i}k_{2i} + q_{2i+1}k_{2i+1}) + j(q_{2i+1}k_{2i} - q_{2i}k_{2i+1})
$$
The real part is the dot product of the original pair, and the imaginary part is related to their cross-product.

Now, we multiply by `e^{j(m-n)\theta_i} = \cos((m-n)\theta_i) + j\sin((m-n)\theta_i)` and take the real part of the result.
Using the rule `Re((A+jB)(C+jD)) = AC - BD`:
$$
\text{PairContribution}_i = \underbrace{(q_{2i}k_{2i} + q_{2i+1}k_{2i+1})}_{\text{Original Dot Product}} \cos((m-n)\theta_i) - \underbrace{(q_{2i+1}k_{2i} - q_{2i}k_{2i+1})}_{\text{Original Cross Product}} \sin((m-n)\theta_i)
$$

### Summary of the Final Formula

The final attention score between a query `q` at position `m` and a key `k` at position `n` is:
$$
\text{Score}(q_m, k_n) = \sum_{i=0}^{d/2 - 1} \left[ (q_{2i}k_{2i} + q_{2i+1}k_{2i+1}) \cos((m-n)\theta_i) + (q_{2i}k_{2i+1} - q_{2i+1}k_{2i}) \sin((m-n)\theta_i) \right]
$$
*(Note: some derivations show a minus sign on the sin term; it depends on the definition of the complex conjugate relationship but the core principle is identical).*

This formula is the beautiful result of RoPE. It shows, without ambiguity, that the final score is a sum of terms where:
1.  One part depends only on the **original content** of the query and key vectors (`q_{2i}k_{2i} + ...`).
2.  The other part is a trigonometric function that depends only on the **relative position `m-n`**.

The vector's own information provides the "what" (semantic similarity), and the rotation operation, using the external position index `m`, injects the "where" (positional context), resulting in a dot product that is sensitive to relative distance.