from flask import Flask, request, jsonify, render_template, url_for
from flask_mail import *  
from cannyeval import *


app = Flask(__name__)

app.config['MAIL_SERVER']='smtp.gmail.com'  
app.config['MAIL_PORT']=465  
app.config['MAIL_USERNAME'] = 'pranuthota31@gmail.com'  
app.config['MAIL_PASSWORD'] = '$%6&*yhn'  
app.config['MAIL_USE_TLS'] = False  
app.config['MAIL_USE_SSL'] = True  
mail = Mail(app)
SEND_MAIL_TO = app.config['MAIL_USERNAME']
UPLOADS_DIR = r"C:\Users\T.PRANEETH\Desktop\ML Workshop\My_Startup_Ideas\Canny.ai"

@app.route('/test')
def juss_testing():
	return "Juss Testing"
@app.route('/gradeit', methods=["POST"])
def grade():
	global SEND_MAIL_TO
	global UPLOADS_DIR
	evaluator = CannyEval()
	try:
		input_form = request.form
		file = request.files['data json']
		if file.filename != '':
			data_json = json.load(file)
		else:
			data_json = json.load(open(input_form['data json']))
			
			
		SEND_MAIL_TO = input_form["mail id"]
		
	except:
		return "SOME ERROR IN APPLICATION"
	report = evaluator.report_card(data_json=data_json, max_marks=int(input_form['max marks']), relative_marking=eval(input_form['relative marking']), integer_marking=eval(input_form['integer marking']), json_load_version="v2")
	dictToReturn = report.to_json()
	csvToMail = report.to_csv(r"C:\Users\T.PRANEETH\Desktop\ML Workshop\My_Startup_Ideas\Canny.ai\userfiles\report.csv")
	return mail_results() #json.loads(dictToReturn)
	
	
@app.route('/')
def home():
	return render_template('home.html')
def mail_results():
	global SEND_MAIL_TO
	global UPLOADS_DIR
	msg = Message("!!We Are Done Evaluating Your Students' Answers!!", sender = 'pranuthota31@gmail.com', recipients=[SEND_MAIL_TO])  
	msg.body = "Greetings from Canny.works,\n\n\tWe took a look into your students' answer sheets and graded them as cannily as possible. Go on and check for yourself if the time you invested in contacting us was worth it or not. Report in the attachment.\n\n\tDon't forget to feed us back with your comments or compliments\n\nRegards,\nCannyTeam " 
	with app.open_resource(r"C:\Users\T.PRANEETH\Desktop\ML Workshop\My_Startup_Ideas\Canny.ai\userfiles\report.csv") as fp:  
		msg.attach("report.csv","text/csv",fp.read())
		mail.send(msg) 
	return "We have delivered your students' report card to your mail id, please do check it out there!!"
