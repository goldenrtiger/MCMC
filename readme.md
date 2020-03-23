# MCMC


# Dependencies
    - python 3.6
    - tensorflow 2.1.0
    - tensorflow_probability
    - numpy
    - pandas
    - matplotlib

# exmaples
    - hmc in abs_MCMC_example.py
    - tf_hmc in abs_tf_hmc_example.py

# formulas
    > estimate parameters

    > s: nozzle size
    > fm: extrusion speed
    > Vr: printing speed
    > Zn: nozzle offset
    > 
    > observation: [s, fm, Vr, Zn]
    > y: targeted width
    > 
    > f(w|a, b, c, d) = s * fm ** a * Zn ** b * c / Vr ** d 
    > p(w|a, b, c, d) = (s * fm ** a * Zn ** b * c / Vr ** d) 
    > params: [a, b, c, d]
    > 
    > 

# Contact
    Welcome for comments and further discussions.

    Author: Xu Jing
    Email: xj.yixing@hotmail.com
