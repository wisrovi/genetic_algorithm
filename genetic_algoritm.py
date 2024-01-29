"""
    This file contains the implementation of a genetic algorithm to optimize the parameters of a dictionary.
    The dictionary is defined in the base_optimize variable and the parameters to optimize are defined in the
    params_to_optimize variable.
    The evaluate function is used to evaluate the individual and return a tuple with the fitness value.
    The process_individual function is used to process the individual and return a value to be optimized.
    The population begins with 50 individuals and evolves over 100 generations.

    @Author: William Steve Rodriguez Villamizar
    @Date: 2024-01-29
    @Github: https://github.com/wisrovi/genetic_algorithm
    @License: MIT
"""

# pip install deap
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd


class GAParams:
    def __init__(
        self,
        population_size,
        evolution_generations,
        params_to_optimize: dict = {},
    ):
        self.population_size = population_size
        self.evolution_generations = evolution_generations
        self.params_to_optimize = params_to_optimize


class GA:
    def __init__(self, ga_params, evaluate):
        self.ga_params = ga_params
        self.evaluate = evaluate
        self.toolbox = self.create_deap_types()
        self.population = self.toolbox.population(n=self.ga_params.population_size)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.results_df = pd.DataFrame(
            columns=[
                "Generation",
                "Best Fitness",
                "Average Fitness",
                "Std Fitness",
                "Best Individual",
            ]
        )

    def create_deap_types(self):
        base_optimize = self.ga_params.params_to_optimize.copy()

        base_optimize["lower_bound"] = dict()
        base_optimize["upper_bound"] = dict()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(
            "Individual",
            dict,
            fitness=creator.FitnessMax,
            **base_optimize,
        )

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.01, 0.9)
        toolbox.register("attr_bool", random.choice, [True, False])
        toolbox.register("attr_int", random.randint, 30, 50)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.init_individual
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.mate)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def init_individual(self):
        base_optimize = self.ga_params.params_to_optimize

        first_individual = {}
        for key, value in base_optimize.items():
            if value == float:
                first_individual[key] = self.toolbox.attr_float()
            elif value == bool:
                first_individual[key] = self.toolbox.attr_bool()
            elif value == int:
                first_individual[key] = self.toolbox.attr_int()
            else:
                breakpoint()
                raise Exception("Invalid type for parameter")

        return first_individual

    def mate(self, ind1, ind2):
        for key in ind1.keys():
            if isinstance(ind1[key], float):
                alpha = random.uniform(-0.5, 1.5)
                lower_bound, upper_bound = ind1.lower_bound.get(
                    key, 0.0
                ), ind1.upper_bound.get(key, 1.0)
                ind1[key] = max(
                    min((1.0 - alpha) * ind1[key] + alpha * ind2[key], upper_bound),
                    lower_bound,
                )
                ind2[key] = max(
                    min(alpha * ind1[key] + (1.0 - alpha) * ind2[key], upper_bound),
                    lower_bound,
                )
        return ind1, ind2

    def mutate(self, individual):
        # la mutación se hace sobre el mismo individuo,
        # dado que es un diccionario, se puede acceder a sus valores para mutarlos de forma individual cada key
        # se muta con una probabilidad del 10% si es un float, 50% si es un booleano y 20% si es un entero

        for key in individual.keys():
            if key in individual.lower_bound and key in individual.upper_bound:
                lower_bound, upper_bound = (
                    individual.lower_bound[key],
                    individual.upper_bound[key],
                )
                if isinstance(individual[key], float):
                    # si es un float, mutar con una distribución gaussiana del 10%
                    mutated_value = max(
                        min(individual[key] + random.gauss(0, 0.1), upper_bound),
                        lower_bound,
                    )
                    individual[key] = round(mutated_value, 2)
                elif isinstance(individual[key], bool):
                    # si es un booleano, mutar con una probabilidad del 50%
                    individual[key] = (
                        not individual[key]
                        if random.random() < 0.5
                        else individual[key]
                    )
                elif isinstance(individual[key], int):
                    # si es un entero, mutar con una distribución gaussiana del 20%
                    mutated_value = max(
                        min(
                            individual[key] + int(random.gauss(0, 0.2)),
                            upper_bound,
                        ),
                        lower_bound,
                    )
                    individual[key] = mutated_value
        return (individual,)

    def evolve_population(self, name_file: str = "results.csv"):
        print(
            "Generation",
            "\t",
            "Best_Fitness",
            "\t",
            "Average_Fitness",
            "\t",
            "Std_Fitness",
        )
        for gen in range(self.ga_params.evolution_generations):
            algorithms.eaMuPlusLambda(
                self.population,
                self.toolbox,
                mu=10,
                lambda_=50,
                cxpb=0.7,
                mutpb=0.2,
                ngen=1,  # Hacer solo una generación a la vez para poder obtener estadísticas, el mejor individuo y almacenarlo en el Hall of Fame
                stats=self.stats,
                halloffame=self.hof,
                verbose=False,
            )

            fits = [ind.fitness.values[0] for ind in self.population]
            length = len(self.population)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean**2) ** 0.5

            best_individual = self.hof[0]

            print(
                gen + 1,
                "\t",
                max(fits),
                "\t",
                mean,
                "\t",
                std,
            )

            self.results_df.loc[len(self.results_df)] = [
                gen + 1,
                max(fits),
                mean,
                std,
                best_individual,
            ]

            self.results_df.to_csv(name_file, index=False)

        self.results_df.to_csv(name_file, index=False)
        best_individual = self.hof[0]

        return best_individual
