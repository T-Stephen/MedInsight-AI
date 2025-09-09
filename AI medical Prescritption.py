# Run this to clean up existing tunnels:
from pyngrok import ngrok
tunnels = ngrok.get_tunnels()
for tunnel in tunnels:
    ngrok.disconnect(tunnel.public_url)
ngrok.kill()

# Then re-run your main code



import nest_asyncio
import threading
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

nest_asyncio.apply()

app = FastAPI()

# Load Granite Model
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.0-2b-instruct")
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.0-2b-instruct")

class DrugAnalysisRequest(BaseModel):
    drugs: List[str]
    age: int
    medical_conditions: Optional[str] = ""

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
def analyze(request: DrugAnalysisRequest):
    messages = [
        {"role": "user", "content": f"Analyze interactions for: {', '.join(request.drugs)} in a {request.age}-year-old patient with {request.medical_conditions}"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=200)
    result_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return {"granite_analysis": result_text}

def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_api, daemon=True).start()
time.sleep(10)  # Give time for backend to start



# Google Colab Ready IBM Granite Drug Analyzer - Enhanced Version
# Run this complete code in ONE cell in Google Colab

# Install packages first
import subprocess
import sys
import os

def install_and_import():
    """Install required packages"""
    packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'streamlit': 'streamlit',
        'pyngrok': 'pyngrok',
        'requests': 'requests',
        'pandas': 'pandas'
    }

    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"📦 Installing {package_name}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '-q'])

    print("✅ All packages ready!")

print("🔧 Setting up environment...")
install_and_import()

# Now create and run the Streamlit app
streamlit_app_code = '''
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import re

# Configure Streamlit
st.set_page_config(
    page_title="IBM Granite Drug Analyzer - Enhanced",
    page_icon="🧬",
    layout="wide"
)

# Enhanced Drug Database with Alternatives
DRUG_DATABASE = {
    "aspirin": {
        "name": "Aspirin (Acetylsalicylic Acid)",
        "class": "NSAID",
        "uses": "Pain relief, fever reduction, blood thinning",
        "adult_dose": "325-650mg every 4-6 hours (max 4g/day)",
        "elderly_dose": "325mg every 6-8 hours (max 2.6g/day)",
        "pediatric_dose": "10-15mg/kg every 6 hours (>12 years)",
        "side_effects": ["Stomach irritation", "Bleeding risk", "Tinnitus", "Reye's syndrome risk in children"],
        "interactions": ["warfarin", "alcohol", "ibuprofen", "methotrexate"],
        "contraindications": ["Active bleeding", "Severe liver disease", "Children <12 years"],
        "warnings": "Monitor for bleeding, avoid in children under 12",
        "alternatives": {
            "same_class": ["ibuprofen", "naproxen", "diclofenac"],
            "different_class": ["acetaminophen", "celecoxib"],
            "natural": ["willow bark", "turmeric", "ginger"]
        },
        "cost": "Low ($)"
    },
    "warfarin": {
        "name": "Warfarin Sodium",
        "class": "Anticoagulant",
        "uses": "Blood clot prevention, atrial fibrillation",
        "adult_dose": "2-10mg daily (individualized based on INR)",
        "elderly_dose": "Start 1-5mg daily, adjust based on INR",
        "pediatric_dose": "0.2mg/kg daily (adjust per INR)",
        "side_effects": ["Bleeding", "Bruising", "Hair loss", "Purple toe syndrome"],
        "interactions": ["aspirin", "alcohol", "antibiotics", "amiodarone"],
        "contraindications": ["Active bleeding", "Pregnancy", "Recent surgery"],
        "warnings": "Regular INR monitoring required (target 2-3)",
        "alternatives": {
            "same_class": ["rivaroxaban", "apixaban", "dabigatran"],
            "different_class": ["heparin", "enoxaparin"],
            "natural": ["fish oil", "garlic", "ginkgo biloba (caution)"]
        },
        "cost": "Low ($)"
    },
    "metformin": {
        "name": "Metformin",
        "class": "Biguanide Antidiabetic",
        "uses": "Type 2 diabetes management, PCOS",
        "adult_dose": "500-1000mg twice daily with meals (max 2550mg/day)",
        "elderly_dose": "500mg daily initially, adjust based on kidney function",
        "pediatric_dose": "500mg twice daily (≥10 years)",
        "side_effects": ["Nausea", "Diarrhea", "Metallic taste", "Lactic acidosis (rare)"],
        "interactions": ["alcohol", "contrast dyes", "topiramate"],
        "contraindications": ["Severe kidney disease", "Metabolic acidosis", "Dehydration"],
        "warnings": "Monitor kidney function, stop before surgery/contrast studies",
        "alternatives": {
            "same_class": ["phenformin (rarely used)"],
            "different_class": ["glipizide", "sitagliptin", "pioglitazone", "insulin"],
            "natural": ["bitter melon", "cinnamon", "chromium", "berberine"]
        },
        "cost": "Low ($)"
    },
    "lisinopril": {
        "name": "Lisinopril",
        "class": "ACE Inhibitor",
        "uses": "High blood pressure, heart failure, post-MI",
        "adult_dose": "10-20mg once daily (max 80mg/day)",
        "elderly_dose": "5-10mg daily initially",
        "pediatric_dose": "0.07mg/kg once daily (≥6 years)",
        "side_effects": ["Dry cough", "Dizziness", "Hyperkalemia", "Angioedema"],
        "interactions": ["NSAIDs", "potassium supplements", "lithium"],
        "contraindications": ["Pregnancy", "Bilateral renal artery stenosis", "History of angioedema"],
        "warnings": "Monitor potassium and kidney function, avoid in pregnancy",
        "alternatives": {
            "same_class": ["enalapril", "captopril", "ramipril"],
            "different_class": ["losartan", "amlodipine", "metoprolol", "hydrochlorothiazide"],
            "natural": ["hawthorn", "garlic", "hibiscus tea", "CoQ10"]
        },
        "cost": "Low ($)"
    },
    "ibuprofen": {
        "name": "Ibuprofen",
        "class": "NSAID",
        "uses": "Pain relief, inflammation, fever reduction",
        "adult_dose": "200-400mg every 6-8 hours (max 3200mg/day)",
        "elderly_dose": "200mg every 8 hours initially",
        "pediatric_dose": "5-10mg/kg every 6-8 hours (≥6 months)",
        "side_effects": ["Stomach upset", "Kidney problems", "Cardiovascular risk"],
        "interactions": ["warfarin", "ACE inhibitors", "lithium", "methotrexate"],
        "contraindications": ["Active peptic ulcer", "Severe heart failure", "Third trimester pregnancy"],
        "warnings": "Use lowest effective dose, monitor kidney function",
        "alternatives": {
            "same_class": ["naproxen", "aspirin", "diclofenac"],
            "different_class": ["acetaminophen", "celecoxib"],
            "natural": ["turmeric", "willow bark", "boswellia"]
        },
        "cost": "Low ($)"
    },
    "acetaminophen": {
        "name": "Acetaminophen (Paracetamol)",
        "class": "Analgesic/Antipyretic",
        "uses": "Pain relief, fever reduction",
        "adult_dose": "325-650mg every 4-6 hours (max 4g/day)",
        "elderly_dose": "325-500mg every 6 hours (max 3g/day)",
        "pediatric_dose": "10-15mg/kg every 4-6 hours",
        "side_effects": ["Hepatotoxicity (overdose)", "Rare skin reactions"],
        "interactions": ["warfarin", "alcohol", "phenytoin"],
        "contraindications": ["Severe liver disease", "Known hypersensitivity"],
        "warnings": "Monitor total daily dose from all sources, hepatotoxic in overdose",
        "alternatives": {
            "same_class": [],
            "different_class": ["ibuprofen", "aspirin", "naproxen"],
            "natural": ["willow bark", "feverfew", "peppermint oil"]
        },
        "cost": "Very Low ($)"
    },
    "amlodipine": {
        "name": "Amlodipine",
        "class": "Calcium Channel Blocker",
        "uses": "High blood pressure, angina",
        "adult_dose": "5-10mg once daily",
        "elderly_dose": "2.5-5mg daily initially",
        "pediatric_dose": "2.5-5mg daily (≥6 years)",
        "side_effects": ["Ankle swelling", "Flushing", "Dizziness", "Gum hyperplasia"],
        "interactions": ["simvastatin", "cyclosporine", "tacrolimus"],
        "contraindications": ["Cardiogenic shock", "Severe aortic stenosis"],
        "warnings": "Monitor for edema, avoid abrupt discontinuation",
        "alternatives": {
            "same_class": ["nifedipine", "diltiazem", "verapamil"],
            "different_class": ["lisinopril", "losartan", "metoprolol"],
            "natural": ["hawthorn", "magnesium", "potassium"]
        },
        "cost": "Low ($)"
    },
    "omeprazole": {
        "name": "Omeprazole",
        "class": "Proton Pump Inhibitor",
        "uses": "GERD, peptic ulcer disease, H. pylori eradication",
        "adult_dose": "20-40mg once daily before meals",
        "elderly_dose": "20mg daily initially",
        "pediatric_dose": "1mg/kg daily (≥1 year)",
        "side_effects": ["Headache", "Diarrhea", "Bone fractures (long-term)", "B12 deficiency"],
        "interactions": ["clopidogrel", "warfarin", "phenytoin"],
        "contraindications": ["Known hypersensitivity"],
        "warnings": "Long-term use may increase infection risk, bone fractures",
        "alternatives": {
            "same_class": ["pantoprazole", "esomeprazole", "lansoprazole"],
            "different_class": ["ranitidine", "famotidine", "sucralfate"],
            "natural": ["aloe vera", "licorice root", "slippery elm"]
        },
        "cost": "Moderate ($$)"
    }
}

# Medical Conditions Database
CONDITIONS_DATABASE = {
    "diabetes": {
        "preferred_drugs": ["metformin", "sitagliptin", "insulin"],
        "avoid_drugs": ["thiazides", "steroids"],
        "monitoring": ["blood glucose", "HbA1c", "kidney function"]
    },
    "hypertension": {
        "preferred_drugs": ["lisinopril", "amlodipine", "hydrochlorothiazide"],
        "avoid_drugs": ["NSAIDs", "decongestants"],
        "monitoring": ["blood pressure", "kidney function", "potassium"]
    },
    "heart_disease": {
        "preferred_drugs": ["aspirin", "lisinopril", "metoprolol"],
        "avoid_drugs": ["NSAIDs", "calcium channel blockers"],
        "monitoring": ["heart rate", "blood pressure", "cardiac function"]
    },
    "kidney_disease": {
        "preferred_drugs": ["furosemide", "erythropoietin"],
        "avoid_drugs": ["NSAIDs", "metformin", "ACE inhibitors"],
        "monitoring": ["creatinine", "potassium", "phosphorus"]
    },
    "liver_disease": {
        "preferred_drugs": ["lactulose", "rifaximin"],
        "avoid_drugs": ["acetaminophen", "NSAIDs", "statins"],
        "monitoring": ["liver enzymes", "bilirubin", "INR"]
    }
}

# IBM Granite Model Class (unchanged as requested)
class GraniteAnalyzer:
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.0-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load_model(self):
        """Load IBM Granite model"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Initializing IBM Granite 3.0...")
            progress_bar.progress(20)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            progress_bar.progress(50)

            status_text.text("🔄 Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            progress_bar.progress(80)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            progress_bar.progress(100)
            status_text.text("✅ IBM Granite model loaded successfully!")

            self.is_loaded = True
            st.success(f"🤖 IBM Granite 3.0 2B ready on {self.device}!")
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("💡 Continuing with rule-based analysis...")
            self.is_loaded = False

    def get_alternative_recommendations(self, current_drugs, age, conditions, allergies=""):
        """Get alternative drug recommendations using IBM Granite"""
        if not self.is_loaded:
            return self._rule_based_alternatives(current_drugs, age, conditions, allergies)

        try:
            # Create detailed prompt for alternatives
            drug_info = []
            for drug in current_drugs:
                if drug.lower() in DRUG_DATABASE:
                    info = DRUG_DATABASE[drug.lower()]
                    drug_info.append(f"{info['name']} - {info['uses']}")
                else:
                    drug_info.append(drug)

            drug_list = "; ".join(drug_info)

            prompt = f"""<|user|>
As a clinical pharmacist, provide alternative medication recommendations:

PATIENT PROFILE:
- Age: {age} years old
- Current Medications: {drug_list}
- Medical Conditions: {conditions or 'None specified'}
- Known Allergies: {allergies or 'None reported'}

Please provide:
1. Alternative medications for each current drug
2. Dosage recommendations based on age
3. Reasons for the alternatives (efficacy, safety, cost)
4. Special considerations for this patient
5. Monitoring requirements

Format as a clear recommendation with rationale.
<|assistant|>"""

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1200, truncation=True)

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            st.warning(f"AI generation failed: {str(e)}")
            return self._rule_based_alternatives(current_drugs, age, conditions, allergies)

    def _rule_based_alternatives(self, drugs, age, conditions, allergies=""):
        """Enhanced rule-based alternative recommendations"""
        analysis = f"🔄 ALTERNATIVE MEDICATION RECOMMENDATIONS\\n"
        analysis += f"Patient: {age} years old ({'Elderly' if age >= 65 else 'Adult'})\\n"
        analysis += f"Conditions: {conditions or 'None specified'}\\n"
        analysis += f"Allergies: {allergies or 'None reported'}\\n\\n"

        for drug in drugs:
            drug_info = DRUG_DATABASE.get(drug.lower())
            if drug_info:
                analysis += f"💊 CURRENT: {drug_info['name']}\\n"
                analysis += f"   Uses: {drug_info['uses']}\\n"

                # Age-appropriate dosing
                if age >= 65:
                    analysis += f"   Current Dose: {drug_info['elderly_dose']}\\n"
                elif age < 18:
                    analysis += f"   Current Dose: {drug_info.get('pediatric_dose', 'Consult pediatrician')}\\n"
                else:
                    analysis += f"   Current Dose: {drug_info['adult_dose']}\\n"

                analysis += f"\\n🔄 ALTERNATIVES:\\n"

                # Same class alternatives
                if drug_info['alternatives']['same_class']:
                    analysis += f"   Same Class Options:\\n"
                    for alt in drug_info['alternatives']['same_class'][:2]:
                        alt_info = DRUG_DATABASE.get(alt.lower())
                        if alt_info:
                            dose_key = "elderly_dose" if age >= 65 else "adult_dose"
                            analysis += f"   • {alt_info['name']} - {alt_info[dose_key]}\\n"

                # Different class alternatives
                if drug_info['alternatives']['different_class']:
                    analysis += f"   Different Class Options:\\n"
                    for alt in drug_info['alternatives']['different_class'][:2]:
                        alt_info = DRUG_DATABASE.get(alt.lower())
                        if alt_info:
                            dose_key = "elderly_dose" if age >= 65 else "adult_dose"
                            analysis += f"   • {alt_info['name']} - {alt_info[dose_key]}\\n"

                # Natural alternatives
                if drug_info['alternatives']['natural']:
                    analysis += f"   Natural Options (consult healthcare provider):\\n"
                    for natural in drug_info['alternatives']['natural'][:2]:
                        analysis += f"   • {natural}\\n"

                analysis += f"   Cost: {drug_info['cost']}\\n\\n"

        # Age-specific considerations
        if age >= 65:
            analysis += f"👴 ELDERLY CONSIDERATIONS:\\n"
            analysis += "• Start with lower doses and titrate slowly\\n"
            analysis += "• Monitor for increased drug sensitivity\\n"
            analysis += "• Consider drug-drug interactions more carefully\\n"
            analysis += "• Regular medication reviews recommended\\n\\n"
        elif age < 18:
            analysis += f"👶 PEDIATRIC CONSIDERATIONS:\\n"
            analysis += "• Weight-based dosing required\\n"
            analysis += "• Some medications contraindicated\\n"
            analysis += "• Consider liquid formulations\\n"
            analysis += "• Parent/caregiver education essential\\n\\n"

        # Condition-based recommendations
        if conditions:
            condition_lower = conditions.lower()
            for condition, info in CONDITIONS_DATABASE.items():
                if condition in condition_lower:
                    analysis += f"🏥 {condition.upper()} CONSIDERATIONS:\\n"
                    analysis += f"• Preferred drugs: {', '.join(info['preferred_drugs'])}\\n"
                    analysis += f"• Avoid: {', '.join(info['avoid_drugs'])}\\n"
                    analysis += f"• Monitor: {', '.join(info['monitoring'])}\\n\\n"

        analysis += "⚠️ Always consult healthcare providers before making medication changes"
        return analysis

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return GraniteAnalyzer()

# Main App
def main():
    # Header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>🧬 IBM Granite Drug Analyzer - Enhanced</h1>
        <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 18px;'>AI-Powered Alternative Medicine Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    analyzer = get_analyzer()

    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Model")

        if not analyzer.is_loaded:
            if st.button("🚀 Load IBM Granite Model", type="primary"):
                analyzer.load_model()
                st.rerun()
        else:
            st.success("✅ IBM Granite Loaded")
            st.info(f"Device: {analyzer.device}")

        st.markdown("---")
        st.header("👤 Patient Profile")

        # Patient Information Form
        with st.form("patient_form"):
            age = st.number_input("Patient Age", 1, 120, 45)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            conditions = st.text_area("Medical Conditions", placeholder="e.g., diabetes, hypertension, heart disease")
            allergies = st.text_area("Known Allergies", placeholder="e.g., penicillin, sulfa drugs")

            st.form_submit_button("💾 Save Profile")

        st.markdown("---")
        st.header("💊 Drug Categories")

        categories = {
            "Pain Relief": ["aspirin", "ibuprofen", "acetaminophen"],
            "Heart/BP": ["lisinopril", "amlodipine", "warfarin"],
            "Diabetes": ["metformin"],
            "Stomach": ["omeprazole"]
        }

        for category, drugs in categories.items():
            with st.expander(f"🏥 {category}"):
                for drug in drugs:
                    st.write(f"• {drug.title()}")

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("💊 Current Medications")

        # Current medications input
        st.subheader("Enter Current Drugs")
        selected_drugs = st.multiselect(
            "Select from database:",
            [drug.title() for drug in DRUG_DATABASE.keys()]
        )

        custom_drug = st.text_input("Or type drug name:")
        if st.button("➕ Add Custom Drug") and custom_drug:
            if custom_drug.title() not in selected_drugs:
                selected_drugs.append(custom_drug.title())
                st.success(f"Added: {custom_drug}")
            else:
                st.warning("Drug already added")

        # Display current medications
        if selected_drugs:
            st.subheader("📋 Current Medication List")
            for i, drug in enumerate(selected_drugs, 1):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{i}. **{drug}**")
                with col_b:
                    if st.button("🗑️", key=f"remove_{i}"):
                        selected_drugs.remove(drug)
                        st.rerun()

        # Quick test scenarios
        st.markdown("---")
        st.subheader("🧪 Quick Test Scenarios")

        test_scenarios = {
            "Elderly Cardiac Patient": {
                "drugs": ["Aspirin", "Warfarin", "Lisinopril"],
                "age": 75,
                "conditions": "atrial fibrillation, hypertension"
            },
            "Diabetic with Pain": {
                "drugs": ["Metformin", "Ibuprofen"],
                "age": 55,
                "conditions": "type 2 diabetes"
            },
            "Young Adult Headache": {
                "drugs": ["Acetaminophen"],
                "age": 25,
                "conditions": "migraine headaches"
            }
        }

        for scenario_name, scenario_data in test_scenarios.items():
            if st.button(f"🎯 {scenario_name}", key=scenario_name):
                st.session_state.test_drugs = scenario_data["drugs"]
                st.session_state.test_age = scenario_data["age"]
                st.session_state.test_conditions = scenario_data["conditions"]
                st.rerun()

    with col2:
        st.header("🔬 Alternative Medicine Analysis")

        # Apply test scenario if selected
        if hasattr(st.session_state, 'test_drugs'):
            selected_drugs = st.session_state.test_drugs
            age = st.session_state.test_age
            conditions = st.session_state.test_conditions
            del st.session_state.test_drugs
            del st.session_state.test_age
            del st.session_state.test_conditions

        if selected_drugs:
            # Analysis buttons
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("🧬 Get AI Alternatives", type="primary"):
                    st.session_state.analysis_type = "alternatives"

            with col_b:
                if st.button("⚠️ Check Interactions", type="secondary"):
                    st.session_state.analysis_type = "interactions"

            # Display analysis results
            if hasattr(st.session_state, 'analysis_type'):

                if st.session_state.analysis_type == "alternatives":
                    with st.spinner("🤖 IBM Granite analyzing alternatives..."):
                        alternatives = analyzer.get_alternative_recommendations(
                            [drug.lower() for drug in selected_drugs],
                            age,
                            conditions,
                            allergies
                        )

                    # Display in tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["🧠 AI Recommendations", "💊 Drug Details", "📊 Comparison", "📋 Summary"])

                    with tab1:
                        st.markdown("### 🤖 IBM Granite Alternative Recommendations")
                        st.markdown(f"""
                        <div style='background: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4169E1;'>
                            <pre style='white-space: pre-wrap; font-family: Arial; font-size: 14px;'>{alternatives}</pre>
                        </div>
                        """, unsafe_allow_html=True)

                    with tab2:
                        st.markdown("### 💊 Current Drug Details")
                        for drug in selected_drugs:
                            drug_info = DRUG_DATABASE.get(drug.lower())
                            if drug_info:
                                with st.expander(f"📋 {drug_info['name']}"):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.write(f"**Class:** {drug_info['class']}")
                                        st.write(f"**Uses:** {drug_info['uses']}")
                                        dose_key = "elderly_dose" if age >= 65 else "adult_dose"
                                        st.write(f"**Recommended Dose:** {drug_info[dose_key]}")
                                        st.write(f"**Cost:** {drug_info['cost']}")
                                    with col_b:
                                        st.write("**Side Effects:**")
                                        for effect in drug_info['side_effects']:
                                            st.write(f"• {effect}")
                                        st.write("**Contraindications:**")
                                        for contra in drug_info.get('contraindications', []):
                                            st.write(f"• {contra}")

                                    st.markdown("**Available Alternatives:**")
                                    alt_types = ['same_class', 'different_class', 'natural']
                                    for alt_type in alt_types:
                                        if drug_info['alternatives'][alt_type]:
                                            st.write(f"*{alt_type.replace('_', ' ').title()}:* {', '.join(drug_info['alternatives'][alt_type])}")

                    with tab3:
                        st.markdown("### 📊 Cost & Safety Comparison")

                        # Create comparison table
                        comparison_data = []
                        for drug in selected_drugs:
                            drug_info = DRUG_DATABASE.get(drug.lower())
                            if drug_info:
                                comparison_data.append({
                                    "Drug": drug_info['name'],
                                    "Class": drug_info['class'],
                                    "Cost": drug_info['cost'],
                                    "Major Side Effects": len(drug_info['side_effects']),
                                    "Interactions": len(drug_info['interactions'])
                                })

                        if comparison_data:
                            import pandas as pd
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True)

                        # Safety alerts
                        st.markdown("### ⚠️ Safety Alerts")
                        if age >= 65:
                            st.warning("👴 **Elderly Patient:** Consider dose reductions and increased monitoring")
                        if age < 18:
                            st.warning("👶 **Pediatric Patient:** Age-appropriate formulations and dosing required")

                        # Check for high-risk combinations
                        high_risk_combos = [
                            (["aspirin", "warfarin"], "High bleeding risk"),
                            (["nsaid", "ace_inhibitor"], "Kidney function concern"),
                            (["metformin", "contrast"], "Lactic acidosis risk")
                        ]

                        selected_lower = [drug.lower() for drug in selected_drugs]
                        for combo, risk in high_risk_combos:
                            if all(any(c in drug for drug in selected_lower) for c in combo):
                                st.error(f"🚨 **{risk}** - Review combination carefully")

                    with tab4:
                        st.markdown("### 📋 Clinical Summary")

                        # Patient summary
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Patient Age", f"{age} years")
                        with col_b:
                            st.metric("Risk Category", "High" if age >= 65 or len(selected_drugs) > 3 else "Standard")
                        with col_c:
                            st.metric("Total Drugs", len(selected_drugs))
                        with col_d:
                            total_interactions = 0
                            for drug in selected_drugs:
                                drug_info = DRUG_DATABASE.get(drug.lower())
                                if drug_info:
                                    total_interactions += len(drug_info['interactions'])
                            st.metric("Potential Interactions", total_interactions)

                        # Recommendations summary
                        st.markdown("#### 🎯 Key Recommendations")
                        recommendations = []

                        if age >= 65:
                            recommendations.append("Consider geriatric dosing adjustments")
                        if len(selected_drugs) > 3:
                            recommendations.append("Review for potential polypharmacy issues")
                        if conditions:
                            recommendations.append("Align therapy with medical conditions")

                        recommendations.extend([
                            "Regular medication reviews with healthcare provider",
                            "Monitor for side effects and drug interactions",
                            "Consider therapeutic alternatives if needed",
                            "Ensure patient understanding of medication regimen"
                        ])

                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")

                        # Export options
                        st.markdown("#### 📤 Export Options")

                        # Generate comprehensive report
                        report = f"""IBM GRANITE DRUG ANALYSIS REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
- Age: {age} years
- Weight: {weight if 'weight' in locals() else 'Not specified'} kg
- Medical Conditions: {conditions or 'None specified'}
- Known Allergies: {allergies or 'None reported'}

CURRENT MEDICATIONS:
{chr(10).join([f"• {drug}" for drug in selected_drugs])}

AI ALTERNATIVE RECOMMENDATIONS:
{alternatives}

CLINICAL RECOMMENDATIONS:
{chr(10).join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])}

DISCLAIMER: This analysis is for educational purposes only.
Always consult with healthcare professionals for medical decisions.
"""

                        col_export1, col_export2 = st.columns(2)
                        with col_export1:
                            st.download_button(
                                "📥 Download Full Report",
                                report,
                                f"drug_alternatives_{int(time.time())}.txt",
                                mime="text/plain"
                            )

                        with col_export2:
                            # Create JSON export
                            json_data = {
                                "patient": {
                                    "age": age,
                                    "conditions": conditions,
                                    "allergies": allergies
                                },
                                "current_medications": selected_drugs,
                                "analysis_date": time.strftime('%Y-%m-%d'),
                                "recommendations": alternatives
                            }

                            st.download_button(
                                "📊 Download JSON Data",
                                json.dumps(json_data, indent=2),
                                f"drug_data_{int(time.time())}.json",
                                mime="application/json"
                            )

                elif st.session_state.analysis_type == "interactions":
                    # Interaction analysis (keeping original functionality)
                    with st.spinner("🔍 Checking drug interactions..."):
                        interaction_analysis = analyzer.analyze_interactions(
                            [drug.lower() for drug in selected_drugs],
                            age,
                            conditions
                        )

                    st.markdown("### ⚠️ Drug Interaction Analysis")
                    st.markdown(f"""
                    <div style='background: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #e53e3e;'>
                        <pre style='white-space: pre-wrap; font-family: Arial;'>{interaction_analysis}</pre>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Welcome screen when no drugs selected
            st.info("👆 Select current medications to get AI-powered alternative recommendations")

            st.markdown("### 🌟 New Features")

            features = [
                "🧠 **AI Alternative Recommendations** - Get personalized medication alternatives",
                "👴 **Age-Specific Dosing** - Pediatric, adult, and geriatric considerations",
                "🏥 **Condition-Based Suggestions** - Recommendations based on medical conditions",
                "💰 **Cost Comparison** - Compare medication costs and accessibility",
                "🌿 **Natural Alternatives** - Explore complementary and alternative options",
                "📊 **Comprehensive Analysis** - Detailed safety and efficacy comparisons",
                "📋 **Clinical Reports** - Professional-grade documentation for healthcare providers"
            ]

            for feature in features:
                st.markdown(feature)

            st.markdown("### 🎯 How to Use")
            steps = [
                "1. **Load IBM Granite Model** - Click the button in the sidebar",
                "2. **Enter Patient Information** - Age, conditions, allergies",
                "3. **Select Current Medications** - Choose from database or add custom",
                "4. **Get AI Recommendations** - Click 'Get AI Alternatives'",
                "5. **Review Alternatives** - Explore different medication options",
                "6. **Download Reports** - Save analysis for healthcare provider"
            ]

            for step in steps:
                st.markdown(step)

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px;'>
        <h4>⚠️ Important Medical Disclaimer</h4>
        <p><strong>This tool is for educational and informational purposes only.</strong></p>
        <p>• Always consult qualified healthcare professionals before making medication changes<br>
        • Do not stop or start medications without medical supervision<br>
        • Individual responses to medications may vary<br>
        • Emergency situations require immediate medical attention</p>

        <hr style='margin: 20px 0;'>

        <p><strong>🤖 Powered by:</strong> IBM Granite 3.0 2B Instruct | Streamlit | Enhanced Drug Database</p>
        <p><strong>📊 Database:</strong> {len(DRUG_DATABASE)} medications with comprehensive alternatives</p>
    </div>
    """, unsafe_allow_html=True)

    # Add medication interaction checker as separate tool
    with st.expander("🔍 Quick Interaction Checker"):
        st.markdown("### Check specific drug combinations")

        drug1 = st.selectbox("First Drug", [""] + list(DRUG_DATABASE.keys()))
        drug2 = st.selectbox("Second Drug", [""] + list(DRUG_DATABASE.keys()))

        if drug1 and drug2 and drug1 != drug2:
            drug1_info = DRUG_DATABASE.get(drug1)
            drug2_info = DRUG_DATABASE.get(drug2)

            interaction_found = False
            if drug1_info and drug2 in drug1_info.get('interactions', []):
                st.warning(f"⚠️ **Potential Interaction Found**")
                st.write(f"**{drug1_info['name']}** may interact with **{drug2_info['name'] if drug2_info else drug2}**")
                interaction_found = True

            if drug2_info and drug1 in drug2_info.get('interactions', []):
                if not interaction_found:
                    st.warning(f"⚠️ **Potential Interaction Found**")
                st.write(f"**{drug2_info['name']}** may interact with **{drug1_info['name'] if drug1_info else drug1}**")
                interaction_found = True

            if not interaction_found:
                st.success("✅ No major interactions found in database")

            st.info("💡 For comprehensive analysis, use the main AI analyzer above")

if __name__ == "__main__":
    main()
'''

# Save the Streamlit app to a file
with open('enhanced_granite_drug_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("📄 Enhanced Streamlit app saved as 'enhanced_granite_drug_app.py'")

# Setup for Colab
print("🔧 Setting up for Google Colab...")

# Install pyngrok and setup
try:
    from pyngrok import ngrok
    import threading
    import subprocess
    import time

    # Kill any existing processes
    try:
        subprocess.run(['pkill', '-f', 'streamlit'], capture_output=True)
    except:
        pass

    # Function to run Streamlit
    def run_streamlit():
        subprocess.run([
            'streamlit', 'run', 'enhanced_granite_drug_app.py',
            '--server.port', '8501',
            '--server.headless', 'true',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false'
        ])

    print("🚀 Starting Enhanced Streamlit App...")
    # Start Streamlit in background
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()

    # Wait for Streamlit to start
    print("⏳ Waiting for Streamlit to initialize...")
    time.sleep(10)

    # Create public tunnel
    print("🌐 Creating public URL with ngrok...")
    try:
        # You can uncomment this line and add your ngrok token for more stable URLs
        # ngrok.set_auth_token("your_token_here")

        public_url = ngrok.connect(8501)

        print("=" * 70)
        print("🎉 SUCCESS! Your Enhanced IBM Granite Drug Analyzer is LIVE!")
        print("=" * 70)
        print(f"🌐 PUBLIC URL: {public_url}")
        print(f"🔗 Click the URL above to access your enhanced app")
        print("=" * 70)
        print("📋 NEW FEATURES:")
        print("✨ AI-powered alternative medicine recommendations")
        print("👴 Age-specific dosing (pediatric, adult, elderly)")
        print("🏥 Condition-based drug suggestions")
        print("💰 Cost comparison and accessibility info")
        print("🌿 Natural and complementary alternatives")
        print("📊 Comprehensive safety comparisons")
        print("📋 Professional clinical reports")
        print("🔍 Quick interaction checker")
        print("=" * 70)
        print("📋 HOW TO USE:")
        print("1. Click the public URL above")
        print("2. Click 'Load IBM Granite Model' (takes 2-3 minutes)")
        print("3. Enter patient age, conditions, and allergies")
        print("4. Select current medications")
        print("5. Click 'Get AI Alternatives' for recommendations")
        print("6. Review detailed analysis in tabs")
        print("7. Download clinical reports")
        print("=" * 70)
        print("🧪 TRY THESE ENHANCED SCENARIOS:")
        print("• Elderly Cardiac Patient: Age 75, multiple heart drugs")
        print("• Diabetic with Pain: Age 55, diabetes + pain management")
        print("• Young Adult: Age 25, simple headache treatment")
        print("=" * 70)

        # Keep the server running
        print("✅ Enhanced app is running! Keep this cell running to maintain access.")
        print("🔄 If URL stops working, just run this cell again.")

    except Exception as e:
        print(f"⚠️ Ngrok setup failed: {e}")
        print("📱 Enhanced app is running locally on port 8501")
        print("💡 For public access:")
        print("1. Get free ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("2. Uncomment and add token in the code above")
        print("3. Run this cell again")

except ImportError:
    print("❌ Pyngrok not installed. Installing now...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyngrok'])
    print("✅ Please run this cell again after installation")

# Instructions for manual setup if auto-setup fails
print("\n" + "=" * 70)
print("🛠️ MANUAL SETUP (if auto-setup fails):")
print("=" * 70)
print("If the above doesn't work, run these commands manually:")
print("1. !streamlit run enhanced_granite_drug_app.py --server.port 8501 &")
print("2. from pyngrok import ngrok; ngrok.connect(8501)")
print("=" * 70)
print("💡 ENHANCEMENT SUMMARY:")
print("• Enhanced drug database with 8+ medications")
print("• Alternative recommendations with rationale")
print("• Age-specific dosing calculations")
print("• Natural and complementary options")
print("• Cost and accessibility comparisons")
print("• Comprehensive clinical reporting")
print("• Quick interaction checker tool")
print("• Professional-grade analysis output")
print("=" * 70)

