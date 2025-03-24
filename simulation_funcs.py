import tqdm
import bokeh
import numpy as np

# Column 0 is change in m, column 1 is change in p
simple_update = np.array([[1, 0, 0, 0],    # Make Healthy mRNA transcript
                        [0, 1, 0, 0],      # Make Damaged mRNA transcript
                        [-1, 0, 0, 0],     # Degrade Healthy mRNA transcript
                        [0, -1, 0, 0],     # Degrade Damaged mRNA transcript
                        [0, 0, 1, 0],      # Accumulate damage on gene
                        [0, 0, -1, 0],     # Repair Damage on gene
                        [0, 1, 0, 0],      # Do not detect damage and make damaged transcript
                        [0, 0, 0, 1],      # Gene turns on
                        [0, 0, 0, -1]      # Gene turns off
                        ], 
                        dtype=int)
                        
def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()
    
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1 # This is returning the reaction event that will occur

def simple_propensity(propensities, population, t, beta_m, gamma_m, u, Pd, gene_on, gene_off):
    """Updates an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack population
    m_healthy = population[0] # number of healthy mRNAs
    m_damaged = population[1] # number of damaged mRNAs
    d = population[2]         # number of damage sites
    G_state = population[3]   # 1 or 0 for gene being ON or OFF

    # Update propensities
    # these events rely on the gene being ON or OFF
    if G_state == 1:  # Gene is ON  
        # Update propensities
        if d == 0: # Gene is ON and has no damage

            propensities[0] = beta_m      # Make healthy mRNA transcript at rate beta_m if no damage on gene
            propensities[1] = 0           # Probability of Make damaged mRNA
            
            propensities[2] = m_healthy * gamma_m # Degrade healthy mRNA
            propensities[3] = m_damaged * gamma_m # Degrade damaged mRNA
            propensities[4] = u           # Add a damage site to the gene
            propensities[5] = 0 # Detect damage site and repair
            propensities[6] = 0 # Do not detect damage site and transcribe mRNA (with error in it)
        
        else: # Gene is ON and has damage 

            propensities[0] = 0      # if there is damage on DNA, probability of making healthy mRNA = 0
            propensities[1] = 0      # Make damaged mRNA

            propensities[2] = m_healthy * gamma_m # Degrade healthy mRNA
            propensities[3] = m_damaged * gamma_m # Degrade damaged mRNA

            propensities[4] = u           # Add a damage site to the gene
            propensities[5] = beta_m * ((1-(1-Pd)**d)) # Detect damage site and repair
            propensities[6] = beta_m * ((1-Pd)**d) # Do not detect damage site and transcribe mRNA (with error in it)

    else: # Gene is OFF (damage is irrelevant here)
        propensities[0] = 0 # no healthy mRNA production
        propensities[1] = 0 # no damaged mRNA production
        propensities[2] = m_healthy * gamma_m # Degrade healthy mRNA
        propensities[3] = m_damaged * gamma_m # Degrade damaged mRNA
        propensities[4] = u           # Add a damage site to the gene
        propensities[5] = 0 # no damage repair
        propensities[6] = 0 # no damage is missed / no damaged mRNA production
    
    propensities[7] = gene_on * (1 - G_state) # Gene turns ON, this will be zero if gene is already ON
    propensities[8] = gene_off * G_state      # Gene turns OFF, this will be zero if gene is already OFF
    
def gillespie_draw(propensity_func, propensities, population, t, args=()):
    """
    Draws a reaction and the time it took to do that reaction.
    
    Parameters
    ----------
    propensity_func : function
        Function with call signature propensity_func(population, t, *args)
        used for computing propensities. This function must return
        an array of propensities.
    population : ndarray
        Current population of particles
    t : float
        Value of the current time.
    args : tuple, default ()
        Arguments to be passed to `propensity_func`.
        
    Returns
    -------
    rxn : int
        Index of reaction that occured.
    time : float
        Time it took for the reaction to occur.
    """
    # Compute propensities
    propensity_func(propensities, population, t, *args)
    
    # Sum of propensities
    props_sum = propensities.sum()
    
    # Compute next time
    time = np.random.exponential(1.0 / props_sum)
    
    # Compute discrete probabilities of each reaction
    rxn_probs = propensities / props_sum
    
    # Draw reaction from this distribution
    rxn = sample_discrete(rxn_probs)
    
    return rxn, time



def gillespie_ssa(propensity_func, update, population_0, time_points, args=()):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from probability distribution of particle counts over time.
    
    Parameters
    ----------
    propensity_func : function
        Function of the form f(params, t, population) that takes the current
        population of particle counts and return an array of propensities
        for each reaction.
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    args : tuple, default ()
        The set of parameters to be passed to propensity_func.        

    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].
    """

    # Initialize output
    pop_out = np.empty((len(time_points), 4), dtype=int) # the length always stays the same as the time points, but the second parameterr
                                                         # must match the amount of variables we are keeping track of

    # Initialize and perform simulation
    i_time = 1
    i = 0
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    propensities = np.zeros(update.shape[0])
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw(propensity_func, propensities, population, t, args)
                
            # Update the population
            population_previous = population.copy()
            population += update[event]

            # can't have negative damage on the gene
            if population[2] < 0: # this needs to match whatever index I am keeping track of the damage at in population
                population[2] = 0
            
            # Increment time
            t += dt

        # Update the index
        i = np.searchsorted(time_points > t, True)
        
        # Update the population
        pop_out[i_time:min(i,len(time_points))] = population_previous
        
        # Increment index
        i_time = i
                        
    return pop_out 