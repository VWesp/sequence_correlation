import argparse
import numpy as np
import pandas as pd
import pyomo.environ as pyo


amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Compute amino acid distributions in proteomes")
	parser.add_argument("-f", "--file", help="Set the path to the distribution file of a domain", required=True)
	args = parser.parse_args()

	file = args.file

	dis_df = pd.read_csv(file, sep="\t", header=0, index_col=0)
	
	f = np.array(dis_df[amino_acids].iloc[50])
	
	m = pyo.ConcreteModel()
	m.I = pyo.RangeSet(0, 19)

	m.x = pyo.Var(m.I, domain=pyo.Integers, bounds=(1, None))
	m.T = pyo.Var(domain=pyo.Integers, bounds=(61, 63))

	m.p = pyo.Var(m.I, domain=pyo.NonNegativeReals)
	m.n = pyo.Var(m.I, domain=pyo.NonNegativeReals)

	m.total = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 61)

	def dev_rule(m, i):
		return m.x[i] - f[i]*m.T == m.p[i] - m.n[i]

	m.dev = pyo.Constraint(m.I, rule=dev_rule)

	m.obj = pyo.Objective(expr=sum(m.p[i] + m.n[i] for i in m.I), sense=pyo.minimize)

	solver = pyo.SolverFactory("highs")
	result = solver.solve(m)
	
	print("Status:", result.solver.status)
	print("Termination:", result.solver.termination_condition)
	print("Objective value:", pyo.value(m.obj))
	print("Sum x:", sum(pyo.value(m.x[i]) for i in m.I))

	for i in m.I:
	    print(amino_acids[i], f[i], int(round(pyo.value(m.x[i]))))