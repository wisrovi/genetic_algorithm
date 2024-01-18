from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
import multiprocessing


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


if __name__ == "__main__":
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

    # Example usage:
    def process_individual(ind):
        values = [float(v) for v in ind.values()]
        return sum(values)

    def evaluate(ind):
        data = process_individual(ind)
        difference = abs(100 - data)
        percentage_closeness = max(0, 1 - (difference / 100))
        return (int(percentage_closeness * 100),)

    ga_params = GAParams(
        population_size=50,
        evolution_generations=100,
        params_to_optimize=base_optimize,
    )
    ga_instance = GA(ga_params, evaluate)
    best_individual = ga_instance.evolve_population()

    print("Best Individual:", best_individual)
    print(process_individual(best_individual))


# exit()


# # https://towardsdatascience.com/intro-to-evolutionary-computation-using-deap-618ca974b8cb

# from deap import base, creator, tools, algorithms
# import random

# import numpy as np
# import pandas as pd

# # Definir el tipo de individuo y población
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create(
#     "Individual",
#     dict,
#     fitness=creator.FitnessMax,
#     conf=float,
#     iou=float,
#     retina_masks=bool,
#     half=bool,
#     track_high_thresh=float,
#     track_low_thresh=float,
#     new_track_thresh=float,
#     track_buffer=int,
#     match_thresh=float,
#     proximity_thresh=float,
#     appearance_thresh=float,
#     with_reid=bool,
#     lower_bound=dict(),  # Agrega un diccionario para almacenar límites inferiores
#     upper_bound=dict(),  # Agrega un diccionario para almacenar límites superiores
# )


# # Inicializar herramientas y funciones de DEAP
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, 0.01, 0.9)
# toolbox.register("attr_bool", random.choice, [True, False])
# toolbox.register("attr_int", random.randint, 30, 50)


# # Definir la inicialización del individuo
# def init_individual():
#     return {
#         "conf": toolbox.attr_float(),
#         "iou": toolbox.attr_float(),
#         "retina_masks": toolbox.attr_bool(),
#         "half": toolbox.attr_bool(),
#         "track_high_thresh": toolbox.attr_float(),
#         "track_low_thresh": toolbox.attr_float(),
#         "new_track_thresh": toolbox.attr_float(),
#         "track_buffer": toolbox.attr_int(),
#         "match_thresh": toolbox.attr_float(),
#         "proximity_thresh": toolbox.attr_float(),
#         "appearance_thresh": toolbox.attr_float(),
#         "with_reid": toolbox.attr_bool(),
#     }


# toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# # Definir la función de evaluación
# deseado = 100
# tamano_poblacion = 100
# generaciones_evolucionar = 1000


# def proceso(ind):
#     valores = [float(v) for v in ind.values()]
#     return sum(valores)


# def evaluate(ind):
#     data = proceso(ind)
#     diferencia = abs(deseado - data)
#     porcentaje_cercania = max(0, 1 - (diferencia / deseado))
#     return (int(porcentaje_cercania * 100),)


# toolbox.register("evaluate", evaluate)


# def mate(ind1, ind2):
#     for key in ind1.keys():
#         if isinstance(ind1[key], float):
#             alpha = random.uniform(-0.5, 1.5)
#             lower_bound, upper_bound = ind1.lower_bound.get(
#                 key, 0.0
#             ), ind1.upper_bound.get(key, 1.0)
#             ind1[key] = max(
#                 min((1.0 - alpha) * ind1[key] + alpha * ind2[key], upper_bound),
#                 lower_bound,
#             )
#             ind2[key] = max(
#                 min(alpha * ind1[key] + (1.0 - alpha) * ind2[key], upper_bound),
#                 lower_bound,
#             )
#     return ind1, ind2


# toolbox.register("mate", mate)


# def mutate(individual):
#     for key in individual.keys():
#         if key in individual.lower_bound and key in individual.upper_bound:
#             lower_bound, upper_bound = (
#                 individual.lower_bound[key],
#                 individual.upper_bound[key],
#             )
#             if isinstance(individual[key], float):
#                 mutated_value = max(
#                     min(individual[key] + random.gauss(0, 0.1), upper_bound),
#                     lower_bound,
#                 )
#                 individual[key] = round(mutated_value, 2)
#             elif isinstance(individual[key], bool):
#                 individual[key] = (
#                     not individual[key] if random.random() < 0.5 else individual[key]
#                 )
#     return (individual,)


# toolbox.register("mutate", mutate)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # Crear una población y evolucionarla
# population = toolbox.population(n=tamano_poblacion)
# hof = tools.HallOfFame(1)  # Hall of Fame para almacenar el mejor individuo

# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)

# # Evolucionar la población y almacenar el mejor individuo en el Hall of Fame

# # Crear un DataFrame para almacenar los resultados de cada generación
# columns = [
#     "Generation",
#     "Best Fitness",
#     "Average Fitness",
#     "Std Fitness",
#     "Best Individual",
# ]
# results_df = pd.DataFrame(columns=columns)


# print("Generación", "Mejor Fitness", "Fitness Promedio", "Desviación Estándar")
# for gen in range(generaciones_evolucionar):
#     # Evolucionar la población y almacenar el mejor individuo en el Hall of Fame
#     algorithms.eaMuPlusLambda(
#         population,
#         toolbox,
#         mu=10,
#         lambda_=50,
#         cxpb=0.7,
#         mutpb=0.2,
#         ngen=1,  # Hacer solo una generación a la vez para poder obtener estadísticas, el mejor individuo y almacenarlo en el Hall of Fame
#         stats=stats,
#         halloffame=hof,
#         verbose=False,
#     )

#     # Obtener estadísticas de la población
#     fits = [ind.fitness.values[0] for ind in population]
#     length = len(population)
#     mean = sum(fits) / length
#     sum2 = sum(x * x for x in fits)
#     std = abs(sum2 / length - mean**2) ** 0.5

#     # Obtener el mejor individuo del Hall of Fame
#     best_individuo = hof[0]

#     print(gen + 1, max(fits), mean, std)

#     # Almacenar los resultados de cada generación en el DataFrame
#     results_df.loc[len(results_df)] = [gen + 1, max(fits), mean, std, best_individuo]


# results_df.to_csv("results.csv", index=False)
# mejor_individuo = hof[0]

# # Imprimir el mejor individuo
# print("Mejor Individuo:", mejor_individuo)
# print(proceso(mejor_individuo))

# exit()


# # ind = {
# #     "conf": (0.2, 0.8),
# #     "iou": (0.5, 0.9),
# #     "retina_masks": (False, True),
# #     "half": (False, True),
# #     "track_high_thresh": (0.4, 0.8),
# #     "track_low_thresh": (0.01, 0.15),
# #     "new_track_thresh": (0.55, 0.88),
# #     "track_buffer": (30, 50),
# #     "match_thresh": (0.4, 0.9),
# #     "proximity_thresh": (0.3, 0.7),
# #     "appearance_thresh": (0.1, 0.5),
# #     "with_reid": (False, True),
# # }

# # # Calcular el mayor valor posible de resultado
# # max_resultado = sum(max(rango) for rango in ind.values())

# # # Calcular el menor valor posible de resultado
# # min_resultado = sum(min(rango) for rango in ind.values())

# # print("Mayor Valor Posible de resultado:", max_resultado)
# # print("Menor Valor Posible de resultado:", min_resultado)


# deseado = 60


# # Definir la función de evaluación
# def proceso(ind):
#     valores = [float(v) for v in ind.values()]
#     return sum(valores)


# def evaluate(ind):
#     data = proceso(ind)
#     diferencia = abs(deseado - data)
#     porcentaje_cercania = max(0, 1 - (diferencia / deseado))
#     return int(porcentaje_cercania * 100) / 100


# # Configurar y ejecutar el algoritmo genético
# individuo_ejemplo = {
#     "conf": 0.40,  # 0.2<->0.8
#     "iou": 0.8,  # 0.5<->0.9
#     "retina_masks": True,  # True<->False
#     "half": True,  # True<->False
#     "track_high_thresh": 0.5,  # 0.4<->0.8
#     "track_low_thresh": 0.08,  # 0.01<->0.15
#     "new_track_thresh": 0.75,  # 0.55<->0.88
#     "track_buffer": 40,  # 30<->50
#     "match_thresh": 0.85,  # 0.4<->0.9
#     "proximity_thresh": 0.5,  # 0.3<->0.7
#     "appearance_thresh": 0.25,  # 0.1<->0.5
#     "with_reid": True,  # True<->False
# }

# print("Evaluando individuo de ejemplo...")
# print("Resultado:", evaluate(individuo_ejemplo))
# print("Evaluando la funcion proceso...")
# print("Resultado:", proceso(individuo_ejemplo))


# from deap import base, creator, tools, algorithms
# import random

# # Definir el tipo de individuo y población
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", dict, fitness=creator.FitnessMax, **individuo_ejemplo)

# # Inicializar herramientas y funciones de DEAP
# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, -5.0, 5.0)
# toolbox.register(
#     "individual", tools.initIterate, creator.Individual, toolbox.attr_float
# )
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # Crear una población y evolucionarla
# population = toolbox.population(n=50)
# hof = tools.HallOfFame(1)  # Hall of Fame para almacenar el mejor individuo


# algorithms.eaMuPlusLambda(
#     population,
#     toolbox,
#     mu=10,
#     lambda_=50,
#     cxpb=0.7,
#     mutpb=0.2,
#     ngen=100,
#     stats=None,
#     halloffame=None,
#     verbose=True,
# )

# # Obtener el mejor individuo del Hall of Fame
# mejor_individuo = hof[0]

# # Imprimir el mejor individuo
# print("Mejor Individuo:", mejor_individuo)
