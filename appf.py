from flask import Flask, request, jsonify
from cannyeval import *
app = Flask(__name__)
@app.route('/test')
def juss_testing():
	return "Juss Testing"
@app.route('/gradeit', methods=["POST"])
def grade():
	evaluator = CannyEval()
	try:
		input_form = request.form
		if input_form['data json'] == 'None' :
			data_json = None
		else:
			data_json = input_form['data json']
		report = evaluator.report_card(data_json=data_json, max_marks=int(input_form['max marks']), relative_marking=bool(input_form['relative marking']), integer_marking=bool(input_form['integer marking']), json_load_version=input_form['json load version'])
		dictToReturn = report.to_json()
		return json.loads(dictToReturn)
	except :
		pass


