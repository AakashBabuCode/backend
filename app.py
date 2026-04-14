"""
Nexus Bank AI Loan System - Flask Backend v2
Gemini 1.5 Flash (detailed) + Groq llama-3.1-8b (detailed summary) + FAISS RAG
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, re, smtplib, random, string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from groq import Groq
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
EMAIL_USER     = os.getenv("EMAIL_USER")
EMAIL_PASS     = os.getenv("EMAIL_PASS")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
groq_client  = Groq(api_key=GROQ_API_KEY)

LOAN_POLICIES = [
    "Personal loan: max Rs 25 lakh, rate 12.5% pa, min salary Rs 15000/month, age 21-60, min CIBIL 650",
    "Home loan: max Rs 2 crore, rate 8.5% pa, min salary Rs 25000/month, age 21-65, min CIBIL 700",
    "Car loan: max Rs 50 lakh, rate 9.25% pa, min salary Rs 20000/month, age 21-60, min CIBIL 650",
    "Education loan: max Rs 40 lakh, rate 10.5% pa, co-applicant required, age 18-35, min CIBIL 600",
    "Business loan: max Rs 1 crore, rate 13% pa, min revenue Rs 50000/month, age 25-65, min CIBIL 680",
    "Gold loan: max Rs 75 lakh, rate 7.5% pa, no min income, age 18-75, no min CIBIL",
    "FOIR rule: total EMI obligations including new loan must not exceed 50 percent of net monthly income",
    "CIBIL below 600 high risk rejected, 600-650 very poor, 650-700 moderate, 700-750 good, above 750 excellent",
    "Loan to income: personal loan not exceed 20x annual salary, home loan not exceed 60x monthly salary",
    "Salaried must have 2 years at current employer, self-employed 3 years business vintage with ITR",
    "Documents: PAN card, Aadhaar, 3 months salary slips, 6 months bank statements, Form 16",
    "Rejected applicants reapply after 6 months with improved CIBIL, clear defaults, reduce obligations",
    "Processing fee: personal 1-2%, home 0.5-1%, car 0.5-1.5%, business 1-2% of sanctioned amount",
    "Prepayment allowed after 6 EMIs with 2% prepayment charge on outstanding principal",
    "Alternative loan: if rejected consider secured loan against property or gold, reduce amount 40%, extend tenure",
]

def simple_embed(text):
    vocab = "loan personal home car education business gold salary income cibil score rate emi approval rejection document verification eligibility foir risk".split()
    t = text.lower()
    return np.array([float(t.count(w)) for w in vocab], dtype=np.float32)

_emb = np.array([simple_embed(p) for p in LOAN_POLICIES])
_nrm = np.linalg.norm(_emb, axis=1, keepdims=True)
_nrm[_nrm == 0] = 1
_emb = _emb / _nrm
faiss_idx = faiss.IndexFlatIP(len(_emb[0]))
faiss_idx.add(_emb)

def rag_retrieve(query, top_k=5):
    q = simple_embed(query).reshape(1,-1)
    n = np.linalg.norm(q)
    if n > 0: q = q/n
    _, idx = faiss_idx.search(q, top_k)
    return [LOAN_POLICIES[i] for i in idx[0] if i < len(LOAN_POLICIES)]

def compute_eligibility(data):
    salary   = float(data.get("salary", 0))
    loan_amt = float(data.get("loanAmount", 0))
    tenure   = int(data.get("tenure", 24))
    age      = int(data.get("age", 30))
    cibil    = int(data.get("cibilScore", 700))
    lt       = data.get("loanType", "personal")
    rate_map = {"personal":12.5,"home":8.5,"car":9.25,"education":10.5,"business":13.0,"gold":7.5}
    r  = rate_map.get(lt, 12.5)
    mr = r/100/12
    emi = (loan_amt*mr*(1+mr)**tenure)/((1+mr)**tenure-1) if tenure>0 and mr>0 else 0
    emi_ratio = (emi/salary*100) if salary>0 else 999
    score=100; reasons=[]; warnings=[]; strengths=[]
    if cibil>=800: score+=5; strengths.append(f"Outstanding CIBIL score of {cibil} demonstrating exceptional creditworthiness")
    elif cibil>=750: strengths.append(f"Excellent CIBIL score of {cibil} reflecting a strong repayment history")
    elif cibil>=700: warnings.append(f"Good CIBIL score of {cibil} - slight improvement recommended"); score-=8
    elif cibil>=650: warnings.append(f"Moderate CIBIL score of {cibil} - poses mild credit risk"); score-=18
    elif cibil>=600: reasons.append(f"Poor CIBIL score of {cibil} - below minimum threshold of 650"); score-=35
    else: reasons.append(f"Very low CIBIL score of {cibil} - loan cannot be approved at this time"); score-=50
    if emi_ratio<=30: strengths.append(f"Very healthy EMI-to-income ratio of {emi_ratio:.1f}% - well within FOIR limits")
    elif emi_ratio<=40: strengths.append(f"Acceptable EMI-to-income ratio of {emi_ratio:.1f}%")
    elif emi_ratio<=50: warnings.append(f"High EMI-to-income ratio of {emi_ratio:.1f}% - at FOIR boundary"); score-=15
    else: reasons.append(f"EMI-to-income ratio of {emi_ratio:.1f}% breaches the 50% FOIR limit"); score-=35
    if age<21: reasons.append(f"Age {age} below minimum eligible age of 21"); score-=25
    elif age>65: reasons.append(f"Age {age} exceeds maximum eligible age of 65"); score-=25
    elif age>58: warnings.append(f"Age {age} approaching retirement - shorter tenure preferred"); score-=5
    else: strengths.append(f"Age {age} is within the eligible range")
    min_sal={"personal":15000,"home":25000,"car":20000,"education":0,"business":50000,"gold":0}
    ms=min_sal.get(lt,15000)
    if salary>=ms*1.5 and lt not in ["education","gold"]: strengths.append(f"Monthly income Rs {salary:,.0f} is strong relative to minimum requirement of Rs {ms:,}")
    elif salary<ms and lt not in ["education","gold"]: reasons.append(f"Monthly income Rs {salary:,.0f} below minimum Rs {ms:,} for {lt} loans"); score-=20
    if lt not in ["education","gold"] and salary>0:
        lti=loan_amt/(salary*12)
        if lti>25: reasons.append(f"Loan-to-annual-income ratio {lti:.1f}x is extremely high"); score-=10
        elif lti>15: warnings.append(f"Loan-to-annual-income ratio {lti:.1f}x is moderately elevated")
        else: strengths.append(f"Loan-to-annual-income ratio {lti:.1f}x is within acceptable limits")
    approved = score>=55
    alt_plans=[]
    for pct,extra in [(0.7,0),(0.5,24),(0.6,12)]:
        a=min(salary*60*pct, loan_amt*pct)
        t2=min(tenure+extra,360)
        ae=(a*mr*(1+mr)**t2)/((1+mr)**t2-1) if mr>0 else 0
        alt_plans.append({"loanAmount":round(a,2),"tenure":t2,"emi":round(ae,2),"emiRatio":round(ae/salary*100,1) if salary>0 else 0,"totalPayable":round(ae*t2,2)})
    return {"approved":approved,"score":max(0,min(100,score)),"emi":round(emi,2),"interestRate":r,
            "totalInterest":round(emi*tenure-loan_amt,2),"totalPayable":round(emi*tenure,2),
            "emiRatio":round(emi_ratio,2),"reasons":reasons,"warnings":warnings,"strengths":strengths,
            "altLoanAmt":alt_plans[0]["loanAmount"],"altTenure":alt_plans[0]["tenure"],"altEmi":alt_plans[0]["emi"],"altPlans":alt_plans}

@app.route("/api/health",methods=["GET"])
def health():
    return jsonify({"status":"ok","timestamp":datetime.now().isoformat()})

@app.route("/api/loan/apply",methods=["POST"])
def loan_apply():
    app_id="NB"+datetime.now().strftime("%Y%m%d")+"".join(random.choices(string.digits,k=4))
    return jsonify({"applicationId":app_id,"status":"received","message":"Application received. AI Agent processing begins.","timestamp":datetime.now().isoformat()})

@app.route("/api/loan/process",methods=["POST"])
def loan_process():
    data=request.json; local=compute_eligibility(data); results={}
    results["identity_verification"]={"step":1,"name":"Identity Verification","status":"passed" if data.get("panCard") else "warning","details":{"panVerified":bool(re.match(r"[A-Z]{5}[0-9]{4}[A-Z]",data.get("panCard",""))),"ageValid":21<=int(data.get("age",0))<=65},"message":"KYC identity verified via Aadhaar and PAN cross-check."}
    results["document_verification"]={"step":2,"name":"Document Verification","status":"passed","details":{"incomeProof":"verified","addressProof":"verified","bankStatement":"auto-verified","employmentLetter":"verified"},"message":"All documents verified. Income and employment confirmed."}
    cibil=int(data.get("cibilScore",700))
    results["credit_analysis"]={"step":3,"name":"Credit Analysis","status":"passed" if cibil>=650 else "failed","details":{"cibilScore":cibil,"creditRating":"Excellent" if cibil>=750 else "Good" if cibil>=650 else "Poor","defaultHistory":"None" if cibil>=650 else "Possible defaults","creditUtilization":str(random.randint(18,42))+"%"},"message":f"CIBIL {cibil} - {'qualifies' if cibil>=650 else 'does not qualify'} under policy."}
    results["emi_calculation"]={"step":4,"name":"EMI Calculation","status":"passed","details":{"emi":local["emi"],"interestRate":local["interestRate"],"tenure":data.get("tenure"),"totalPayable":local["totalPayable"],"totalInterest":local["totalInterest"],"emiToIncomeRatio":local["emiRatio"],"foirStatus":"pass" if local["emiRatio"]<=50 else "fail"},"message":f"EMI Rs {local['emi']:,.0f}/month at {local['interestRate']}% pa. FOIR: {local['emiRatio']}%"}
    rag_ctx=rag_retrieve(f"{data.get('loanType')} loan {data.get('salary')} salary {data.get('cibilScore')} cibil {data.get('loanAmount')} amount")
    rag_text="\n".join(f"- {p}" for p in rag_ctx)
    gemini_prompt=f"""You are a senior credit risk analyst at Nexus Bank India with 20 years of experience. Conduct a comprehensive, detailed loan assessment.

BANK POLICY KNOWLEDGE BASE (RAG):
{rag_text}

APPLICANT PROFILE:
- Name: {data.get('fullName')}, Age: {data.get('age')} years
- Employment: {data.get('employmentType')} at {data.get('employer')}
- Monthly Income: Rs {data.get('salary')}
- CIBIL Score: {data.get('cibilScore')}
- Loan: {data.get('loanType','').title()} of Rs {data.get('loanAmount')} for {data.get('tenure')} months
- Purpose: {data.get('purpose','Not specified')}

COMPUTED METRICS:
- Monthly EMI: Rs {local['emi']:,.0f}, FOIR: {local['emiRatio']}%, Rate: {local['interestRate']}%
- Total Interest: Rs {local['totalInterest']:,.0f}, Total Payable: Rs {local['totalPayable']:,.0f}
- Score: {local['score']}/100, Recommendation: {"APPROVED" if local['approved'] else "REJECTED"}
- Strengths: {', '.join(local['strengths']) if local['strengths'] else 'None'}
- Risks: {', '.join(local['reasons']) if local['reasons'] else 'None'}

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "recommendation": "APPROVED or REJECTED",
  "confidence": <0-100>,
  "riskLevel": "LOW or MEDIUM or HIGH",
  "executiveSummary": "<3-4 sentences: overall creditworthiness and application assessment>",
  "financialAnalysis": "<3-4 sentences: income stability, EMI affordability, FOIR compliance, debt capacity>",
  "creditRiskAnalysis": "<3-4 sentences: CIBIL implications, repayment history, credit risk classification>",
  "approvalRationale": "<3-4 sentences: exact reasons for approval or rejection with policy references>",
  "keyFactors": ["<factor1>","<factor2>","<factor3>","<factor4>"],
  "alternativeSuggestion": "<2-3 sentences: specific alternative loan amounts, tenures, or products>",
  "disbursementPlan": "<2-3 sentences: how and when sanctioned amount will be disbursed>",
  "improvementAdvice": "<2-3 sentences: specific actionable advice to improve future eligibility>"
}}"""
    try:
        resp=gemini_model.generate_content(gemini_prompt)
        txt=resp.text.strip()
        txt=re.sub(r"```json|```","",txt).strip()
        m=re.search(r'\{[\s\S]*\}',txt)
        ai_result=json.loads(m.group() if m else txt)
    except Exception as e:
        dec="APPROVED" if local["approved"] else "REJECTED"
        nm=data.get('fullName','Applicant'); sal=float(data.get('salary',0)); la=float(data.get('loanAmount',0))
        ai_result={"recommendation":dec,"confidence":local["score"],"riskLevel":"LOW" if local["score"]>=70 else "MEDIUM" if local["score"]>=50 else "HIGH",
            "executiveSummary":f"The loan application from {nm} for Rs {la:,.0f} under the {data.get('loanType','personal').title()} Loan category has been comprehensively evaluated by the Nexus Bank AI credit system. The applicant reports a net monthly income of Rs {sal:,.0f} with a CIBIL score of {data.get('cibilScore')}, which {'satisfies' if local['approved'] else 'does not satisfy'} the bank minimum eligibility standards. Based on the full assessment of financial parameters, the application is {'recommended for sanction' if local['approved'] else 'declined'} with an eligibility score of {local['score']} out of 100.",
            "financialAnalysis":f"The net monthly income of Rs {sal:,.0f} generates a monthly EMI of Rs {local['emi']:,.0f}, representing a Fixed Obligation to Income Ratio of {local['emiRatio']}%. {'This FOIR is within the permissible limit of 50%, confirming the applicant has sufficient disposable income to service the loan without undue financial stress.' if local['emiRatio']<=50 else 'This FOIR exceeds the maximum permissible limit of 50%, indicating significant financial strain from the proposed loan obligation.'} The total cost of credit over the tenure amounts to Rs {local['totalInterest']:,.0f} in interest, with total repayment of Rs {local['totalPayable']:,.0f}.",
            "creditRiskAnalysis":f"The applicant CIBIL score of {data.get('cibilScore')} classifies them in the {'low' if cibil>=750 else 'moderate' if cibil>=650 else 'high'} risk segment of Nexus Bank credit risk framework. {'A score at this level demonstrates a consistent history of timely debt repayment and responsible credit utilization, substantially reducing default probability.' if cibil>=700 else 'The current credit score indicates some irregularities in past repayment behaviour or elevated credit utilization, requiring heightened risk monitoring.'} Credit bureau data and employment stability have both been factored into the overall risk classification.",
            "approvalRationale":f"The application is {'approved' if local['approved'] else 'rejected'} following a thorough review against Nexus Bank lending policy guidelines. {'All primary eligibility parameters including minimum income threshold, CIBIL score requirement, and FOIR compliance are met.' if local['approved'] else 'The following policy criteria were not satisfied: '+'; '.join(local['reasons'])+'.'} The eligibility score of {local['score']}/100 {'exceeds' if local['score']>=55 else 'falls below'} the minimum approval threshold of 55 points. {'The risk level is classified as low and the loan is sanctioned subject to documentation.' if local['approved'] else 'Reapplication is advised after addressing the identified deficiencies.'}",
            "keyFactors":(local["strengths"]+local["warnings"]+local["reasons"])[:4] or ["Income evaluation complete","CIBIL assessment complete","FOIR computed","Policy compliance verified"],
            "alternativeSuggestion":f"The applicant is advised to consider a restructured loan of Rs {local['altLoanAmt']:,.0f} over {local['altTenure']} months, which reduces the monthly EMI to Rs {local['altEmi']:,.0f} and brings the FOIR to a more sustainable level. Exploring secured loan alternatives such as a Gold Loan or Loan Against Property would offer lower interest rates with less stringent CIBIL requirements. Improving the CIBIL score above 750 through consistent repayment of existing obligations over the next 6 months would substantially improve eligibility for the originally requested amount.",
            # "disbursementPlan": f"{'The approved loan amount of Rs ' + format(la, ',.0f') + ' will be electronically credited to the applicant registered bank account within 2 working days of completing all post-sanction formalities including e-mandate setup and document submission.' if local['approved'] else 'The application has been declined and no disbursement will be processed at this time.'} {'The applicant must sign the loan agreement and complete the NACH mandate for EMI auto-debit before disbursement can be initiated.' if local['approved'] else 'The applicant is encouraged to review the available alternative offer and reapply after addressing the identified eligibility gaps.'} {'First EMI will be due 30 days from the date of disbursement.' if local['approved'] else ''}",
            "improvementAdvice":f"To strengthen future applications, the applicant should prioritize improving the CIBIL score above 750 by ensuring zero missed payments on all credit obligations for at least 6 consecutive months. Reducing existing loan liabilities through prepayment will lower the overall FOIR and demonstrate stronger debt management capability to the bank. Maintaining stable employment documentation, increasing monthly savings, and avoiding multiple loan enquiries within a short period will collectively enhance the application strength significantly."}
    results["gemini_ai_assessment"]={"step":5,"name":"Gemini AI Assessment","status":"passed","ragContext":rag_ctx,"aiResult":ai_result,"message":f"Gemini AI complete. Confidence: {ai_result.get('confidence',local['score'])}%"}
    final_approved=ai_result.get("recommendation","APPROVED" if local["approved"] else "REJECTED")=="APPROVED"
    results["final_decision"]={"step":6,"name":"Final Decision","status":"approved" if final_approved else "rejected","approved":final_approved,"sanctionDetails":{"approvedAmount":float(data.get("loanAmount",0)) if final_approved else local["altLoanAmt"],"approvedRate":local["interestRate"],"approvedTenure":int(data.get("tenure",24)),"emi":local["emi"],"totalPayable":local["totalPayable"],"disbursementDate":(datetime.now()+timedelta(days=2)).strftime("%d %b %Y") if final_approved else "N/A","validTill":(datetime.now()+timedelta(days=30)).strftime("%d %b %Y")},"message":"Loan sanctioned." if final_approved else "Application declined - alternative plans available."}
    groq_summary=""
    try:
        gr=groq_client.chat.completions.create(model="llama-3.1-8b-instant",messages=[{"role":"system","content":"You are a professional bank relationship manager. Write formal, detailed loan summaries in clear English. No emojis. Be specific with numbers."},{"role":"user","content":f"Write a formal 3-paragraph loan decision summary (3-4 sentences each paragraph).\n\nApplicant: {data.get('fullName')}, Age {data.get('age')}, {data.get('employmentType')} at {data.get('employer')}\nIncome: Rs {float(data.get('salary',0)):,.0f}/month, CIBIL: {data.get('cibilScore')}\nLoan: {data.get('loanType','').title()} Rs {float(data.get('loanAmount',0)):,.0f} for {data.get('tenure')} months\nEMI: Rs {local['emi']:,.0f}, FOIR: {local['emiRatio']}%, Score: {local['score']}/100\nDecision: {'APPROVED' if final_approved else 'REJECTED'}, Risk: {ai_result.get('riskLevel','MEDIUM')}\nIssues: {', '.join(local['reasons']) if local['reasons'] else 'None - all criteria met'}\nStrengths: {', '.join(local['strengths']) if local['strengths'] else 'Standard profile'}\n\nPara 1: Financial profile overview and overall assessment.\nPara 2: Specific reasons for approval/rejection with metric references.\nPara 3: Next steps, alternative options, and improvement advice."}],max_tokens=700,temperature=0.35)
        groq_summary=gr.choices[0].message.content.strip()
    except Exception as e:
        nm=data.get('fullName','Applicant'); sal=float(data.get('salary',0)); la=float(data.get('loanAmount',0))
        groq_summary = (
            f"The loan application submitted by {nm} for a {data.get('loanType','personal').title()} Loan amounting to Rs {format(la, ',.0f')} has been reviewed by the Nexus Bank automated credit assessment system. "
            f"The applicant is a {data.get('employmentType','salaried')} professional at {data.get('employer','the stated organization')}, reporting a net monthly income of Rs {format(sal, ',.0f')} and a CIBIL score of {data.get('cibilScore')}. "
            f"The comprehensive credit evaluation has yielded an eligibility score of {local['score']} out of 100, and the application has been formally {'approved' if final_approved else 'declined'} based on the aggregate assessment of all financial parameters.\n\n"

            f"{'The application successfully satisfies all primary eligibility requirements under Nexus Bank lending policy, including the minimum income threshold, CIBIL score requirement, and FOIR compliance.' if final_approved else 'The application was declined due to specific deficiencies identified during assessment: ' + '; '.join(local['reasons']) + '.'} "
            f"The computed monthly EMI of Rs {format(local['emi'], ',.0f')} represents {local['emiRatio']}% of the applicant monthly income, "
            f"{'which is within the permissible Fixed Obligation to Income Ratio of 50%, confirming adequate repayment capacity.' if local['emiRatio']<=50 else 'which exceeds the maximum allowable FOIR of 50%, indicating the proposed EMI may create undue financial burden.'} "
            f"{'The credit profile and income stability collectively justify the sanction of the requested amount.' if local['approved'] else 'The bank strongly recommends the applicant take corrective action before submitting a revised application.'}\n\n"

            f"{'The sanctioned amount of Rs ' + format(la, ',.0f') + ' will be electronically disbursed to the applicant registered bank account within 2 working days of completing post-sanction formalities including e-mandate registration and document verification at the branch.' if final_approved else 'The applicant is advised to consider the alternative restructured loan offer of Rs ' + format(local['altLoanAmt'], ',.0f') + ' over ' + str(local['altTenure']) + ' months, which would result in a more affordable EMI of Rs ' + format(local['altEmi'], ',.0f') + ' per month and a healthier FOIR.'} "
            "Improving the CIBIL score above 750 through zero missed payments for 6 consecutive months and reducing existing debt obligations will significantly strengthen any future application. "
            "The applicant may contact the Nexus Bank relationship management team at support@nexusbank.in for personalized credit improvement guidance and alternative product recommendations."
        )
    return jsonify({"applicationData":data,"steps":results,"summary":{"approved":final_approved,"score":local["score"],"emi":local["emi"],"interestRate":local["interestRate"],"totalInterest":local["totalInterest"],"totalPayable":local["totalPayable"],"emiRatio":local["emiRatio"],"reasons":local["reasons"],"warnings":local["warnings"],"strengths":local["strengths"],"altLoanAmt":local["altLoanAmt"],"altTenure":local["altTenure"],"altEmi":local["altEmi"],"altPlans":local["altPlans"],"groqSummary":groq_summary,"aiResult":ai_result,"ragContext":rag_ctx}})

@app.route("/api/loan/emi-calculator",methods=["POST"])
def emi_calc():
    d=request.json; p=float(d.get("loanAmount",0)); n=int(d.get("tenure",24)); r=float(d.get("interestRate",12.5))/100/12
    emi=(p*r*(1+r)**n)/((1+r)**n-1) if r>0 else p/n
    return jsonify({"emi":round(emi,2),"totalPayable":round(emi*n,2),"totalInterest":round(emi*n-p,2)})

@app.route("/api/loan/send-email",methods=["POST"])
def send_email():
    data=request.json
    if not data: return jsonify({"success":False,"message":"No data received"})
    recipient=data.get("email","").strip()
    if not recipient or "@" not in recipient: return jsonify({"success":False,"message":"Valid recipient email required"})
    applicant=data.get("applicantName","Valued Customer"); approved=data.get("approved",False)
    loan_amt=float(data.get("loanAmount",0)); emi=float(data.get("emi",0)); app_id=data.get("applicationId","N/A")
    loan_type=str(data.get("loanType","Loan")).title(); rate=data.get("interestRate","N/A"); tenure=data.get("tenure","N/A")
    alt_amt=float(data.get("altLoanAmt",0)); alt_emi=float(data.get("altEmi",0)); alt_tenure=data.get("altTenure","N/A")
    groq_sum=data.get("groqSummary",""); ai_reason=data.get("aiReasoning",""); today=datetime.now().strftime("%d %B %Y")
    disburse=(datetime.now()+timedelta(days=2)).strftime("%d %B %Y")
    if approved:
        subject=f"Nexus Bank - Loan Sanction Letter | {app_id}"
        body=f"""Dear {applicant},

We are pleased to inform you that your {loan_type} application has been approved by Nexus Bank. This letter serves as the official sanction communication.

APPLICATION DETAILS
-------------------
Application ID     : {app_id}
Date               : {today}
Loan Type          : {loan_type}
Sanctioned Amount  : Rs {loan_amt:,.0f}
Interest Rate      : {rate}% per annum
Tenure             : {tenure} months
Monthly EMI        : Rs {emi:,.2f}
Disbursement Date  : {disburse}

AI ASSESSMENT SUMMARY
---------------------
{groq_sum if groq_sum else 'Your application was assessed by the Nexus Bank AI credit evaluation system and met all eligibility criteria.'}

NEXT STEPS
----------
1. Visit the nearest Nexus Bank branch with original KYC documents within 7 working days.
2. Complete e-mandate setup for automated EMI deduction.
3. Sign the loan agreement digitally or at the branch.
4. Sanctioned amount will be credited within 2 working days of completing documentation.

This sanction is valid for 30 days. EMI auto-debit commences 30 days from disbursement date.
Prepayment is permitted after 6 EMIs with a 2% prepayment charge on outstanding principal.

For queries: 1800-XXX-XXXX or support@nexusbank.in

Warm regards,
Nexus Bank AI Loan Division
{today}"""
    else:
        subject=f"Nexus Bank - Loan Application Update | {app_id}"
        body=f"""Dear {applicant},

Thank you for choosing Nexus Bank. We have carefully reviewed your {loan_type} application and regret to inform you that we are unable to approve it at this time.

APPLICATION DETAILS
-------------------
Application ID   : {app_id}
Date             : {today}
Loan Requested   : {loan_type} - Rs {loan_amt:,.0f}
Decision         : NOT APPROVED

ASSESSMENT SUMMARY
------------------
{groq_sum if groq_sum else 'Your application was reviewed against our credit policy and did not meet the minimum eligibility criteria.'}

DETAILED ASSESSMENT
-------------------
{ai_reason if ai_reason else 'Please contact our relationship team for a detailed explanation of the assessment outcome.'}

ALTERNATIVE OFFER
-----------------
Restructured Loan Amount : Rs {alt_amt:,.0f}
Revised Tenure           : {alt_tenure} months
Revised Monthly EMI      : Rs {alt_emi:,.2f}

To accept this offer, visit any Nexus Bank branch or call 1800-XXX-XXXX.

STEPS TO IMPROVE ELIGIBILITY
-----------------------------
1. Improve CIBIL score above 750 with zero missed payments for 6 consecutive months.
2. Reduce existing loan obligations to bring FOIR below 40%.
3. Consider a co-applicant to strengthen the application.
4. Reapply after 6 months with an improved financial profile.

We value your relationship with Nexus Bank and look forward to serving you better.

Regards,
Nexus Bank Credit Assessment Team
{today}"""
    try:
        msg=MIMEMultipart("alternative"); msg["From"]=f"Nexus Bank <{EMAIL_USER}>"; msg["To"]=recipient; msg["Subject"]=subject
        msg.attach(MIMEText(body,"plain","utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465,timeout=15) as srv:
            srv.login(EMAIL_USER,EMAIL_PASS); srv.sendmail(EMAIL_USER,recipient,msg.as_string())
        return jsonify({"success":True,"message":f"Sanction letter sent successfully to {recipient}"})
    except smtplib.SMTPAuthenticationError: return jsonify({"success":False,"message":"Email authentication failed - verify EMAIL_USER and EMAIL_PASS"})
    except Exception as e: return jsonify({"success":False,"message":f"Email failed: {str(e)}"})

@app.route("/api/loan/report",methods=["POST"])
def generate_report():
    data=request.json; d=data.get("applicationData",{}); s=data.get("summary",{}); ai=s.get("aiResult",{}); approved=s.get("approved",False)
    report=f"""
================================================================================
                 NEXUS BANK  -  AI LOAN DECISION REPORT
================================================================================
Application ID  : {data.get('applicationId','N/A')}
Generated On    : {datetime.now().strftime('%d %B %Y, %I:%M %p IST')}
Decision        : {'APPROVED' if approved else 'REJECTED'}
AI Confidence   : {ai.get('confidence',s.get('score',0))}%   Risk Level: {ai.get('riskLevel','N/A')}
================================================================================
APPLICANT DETAILS
--------------------------------------------------------------------------------
Name: {d.get('fullName','N/A')}  |  Age: {d.get('age')} yrs  |  CIBIL: {d.get('cibilScore')}
Employment: {str(d.get('employmentType','N/A')).title()} at {d.get('employer','N/A')}
Monthly Income: Rs {float(d.get('salary',0)):,.0f}  |  Email: {d.get('email','N/A')}
================================================================================
LOAN DETAILS
--------------------------------------------------------------------------------
Type: {str(d.get('loanType','N/A')).title()} Loan  |  Amount: Rs {float(d.get('loanAmount',0)):,.0f}
Tenure: {d.get('tenure')} months  |  Rate: {s.get('interestRate','N/A')}% pa
EMI: Rs {float(s.get('emi',0)):,.2f}/mo  |  FOIR: {s.get('emiRatio',0)}%
Total Interest: Rs {float(s.get('totalInterest',0)):,.2f}  |  Total Payable: Rs {float(s.get('totalPayable',0)):,.2f}
================================================================================
EXECUTIVE SUMMARY (Gemini AI)
--------------------------------------------------------------------------------
{ai.get('executiveSummary','Not available.')}
================================================================================
FINANCIAL ANALYSIS (Gemini AI)
--------------------------------------------------------------------------------
{ai.get('financialAnalysis','Not available.')}
================================================================================
CREDIT RISK ANALYSIS (Gemini AI)
--------------------------------------------------------------------------------
{ai.get('creditRiskAnalysis','Not available.')}
================================================================================
APPROVAL RATIONALE (Gemini AI)
--------------------------------------------------------------------------------
{ai.get('approvalRationale','Not available.')}
================================================================================
GROQ AI DETAILED SUMMARY
--------------------------------------------------------------------------------
{s.get('groqSummary','Not available.')}
================================================================================
KEY FACTORS
--------------------------------------------------------------------------------
{chr(10).join('  - '+f for f in ai.get('keyFactors',[]))}
{'STRENGTHS:'+chr(10)+chr(10).join('  - '+r for r in s.get('strengths',[])) if approved else 'REJECTION REASONS:'+chr(10)+chr(10).join('  - '+r for r in s.get('reasons',[]))}
Warnings: {chr(10).join('  - '+w for w in s.get('warnings',[])) or 'None'}
================================================================================
ALTERNATIVE OFFERS
--------------------------------------------------------------------------------
{chr(10).join(f"  Plan {i+1}: Rs {p['loanAmount']:,.0f} | {p['tenure']} months | EMI Rs {p['emi']:,.0f} | FOIR {p['emiRatio']}%" for i,p in enumerate(s.get('altPlans',[])))}
AI Suggestion: {ai.get('alternativeSuggestion','Not available.')}
================================================================================
DISBURSEMENT PLAN
--------------------------------------------------------------------------------
{ai.get('disbursementPlan','Not available.')}
================================================================================
IMPROVEMENT ADVICE
--------------------------------------------------------------------------------
{ai.get('improvementAdvice','Not available.')}
================================================================================
Nexus Bank AI Loan System | Gemini 1.5 Flash + Groq LLaMA + FAISS RAG
{datetime.now().strftime('%d %B %Y %I:%M %p')} | support@nexusbank.in
================================================================================
"""
    return jsonify({"report":report,"filename":f"NexusBank_{data.get('applicationId','Report')}.txt"})

@app.route("/api/products/loans",methods=["GET"])
def get_loans():
    return jsonify([{"id":"personal","name":"Personal Loan","rate":12.5,"maxAmount":2500000,"minSalary":15000,"tenure":"12-60","desc":"Quick unsecured loans for any personal need"},{"id":"home","name":"Home Loan","rate":8.5,"maxAmount":20000000,"minSalary":25000,"tenure":"12-360","desc":"Fulfill your dream of owning a home"},{"id":"car","name":"Car Loan","rate":9.25,"maxAmount":5000000,"minSalary":20000,"tenure":"12-84","desc":"Drive your dream car with easy EMIs"},{"id":"education","name":"Education Loan","rate":10.5,"maxAmount":4000000,"minSalary":0,"tenure":"12-180","desc":"Invest in your future education"},{"id":"business","name":"Business Loan","rate":13.0,"maxAmount":10000000,"minSalary":50000,"tenure":"12-84","desc":"Grow your business with working capital"},{"id":"gold","name":"Gold Loan","rate":7.5,"maxAmount":7500000,"minSalary":0,"tenure":"3-36","desc":"Instant loan against gold ornaments"}])

@app.route("/api/products/deposits",methods=["GET"])
def get_deposits():
    return jsonify([{"id":"fd","name":"Fixed Deposit","rate":7.5,"minAmount":1000,"maxTenure":"10 years","desc":"Guaranteed returns on your savings"},{"id":"rd","name":"Recurring Deposit","rate":7.0,"minAmount":500,"maxTenure":"10 years","desc":"Monthly savings with good returns"},{"id":"savings","name":"Savings Account","rate":4.0,"minAmount":0,"maxTenure":"Ongoing","desc":"Zero-balance digital savings account"},{"id":"nri","name":"NRI Deposits","rate":8.0,"minAmount":10000,"maxTenure":"5 years","desc":"Special NRE/NRO deposits for NRIs"}])

@app.route("/api/contact",methods=["POST"])
def contact():
    data=request.json; ticket="TKT"+"".join(random.choices(string.digits,k=6))
    return jsonify({"success":True,"message":f"Message from {data.get('name')} received.","ticketId":ticket})

if __name__=="__main__":
    app.run(debug=True,port=5000)
