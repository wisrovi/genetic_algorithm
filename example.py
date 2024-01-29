from genetic_algoritm import GA, GAParams


if __name__ == "__main__":
    """
    This example shows how to use the genetic algorithm to optimize the parameters of a dictionary.
    """

    base_optimize = dict(
        conf=float,
        iou=float,
        retina_masks=bool,
        half=bool,
        track_high_thresh=float,
        track_low_thresh=float,
        new_track_thresh=float,
        track_buffer=int,
        match_thresh=float,
        proximity_thresh=float,
        appearance_thresh=float,
        with_reid=bool,
    )

    def process_individual(ind: dict) -> float:
        """
        This function is used to process the individual and return a value to be optimized.

        @param ind: The individual to be processed.
        @return: A value to be optimized.
        """
        values = [float(v) for v in ind.values()]
        return sum(values)

    def evaluate(ind: dict) -> tuple:
        """
        This function is used to evaluate the individual and return a tuple with the fitness value.

        @param ind: The individual to be evaluated.
        @return: A tuple with the fitness value.
        """
        data = process_individual(ind)
        difference = abs(100 - data)
        percentage_closeness = max(0, 1 - (difference / 100))
        return (int(percentage_closeness * 100),)

    """
    In this example, the genetic algorithm will try to find the best individual that is closest to the value of 100.

    The population also begins with 50 individuals and evolves over 100 generations.
    """
    ga_params = GAParams(
        population_size=50,
        evolution_generations=100,
        params_to_optimize=base_optimize,
    )

    # Create the genetic algorithm instance
    ga_instance = GA(ga_params, evaluate)

    # Evolve the population
    best_individual = ga_instance.evolve_population()

    # Print the best individual, this will be the individual with the highest fitness value.
    print("Best Individual:", best_individual)

    # Evaluate the best individual to find the optimized value.
    print(process_individual(best_individual))
