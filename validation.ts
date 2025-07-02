// frontend/src/utils/medical/validation.ts
// Utilitários de validação médica para CardioAI Pro

// Tipos e interfaces
export interface ECGValidationResult {
  valid: boolean;
  quality: number;
  issues: string[];
  metrics: {
    snr: number;
    baselineWander: number;
    completeness: number;
  };
}

export interface HeartRateValidationResult {
  valid: boolean;
  category: 'normal' | 'bradycardia' | 'severe_bradycardia' | 'tachycardia' | 'severe_tachycardia' | 'athletic_bradycardia';
  warnings: string[];
  alerts: string[];
  errors: string[];
  ageAdjusted: boolean;
  notes: string[];
}

export interface BloodPressureValidationResult {
  valid: boolean;
  category: 'normal' | 'elevated' | 'hypertension' | 'hypertensive_crisis' | 'hypotension';
  stage?: 'optimal' | 'stage_1' | 'stage_2';
  warnings: string[];
  alerts: string[];
  errors: string[];
  urgent?: boolean;
  map: number;
  pulsePressure: number;
  pulsePressureStatus: 'normal' | 'narrow' | 'wide';
}

export interface QTValidationResult {
  valid: boolean;
  qtc: number;
  category: 'normal' | 'borderline' | 'prolonged' | 'severely_prolonged' | 'short';
  risk: 'low' | 'moderate' | 'high' | 'critical';
  alerts: string[];
  warnings: string[];
  formula: 'Bazett' | 'Fridericia' | 'Framingham' | 'Hodges';
  drugInducedRisk?: boolean;
  recommendations: string[];
  deltaQTc?: number;
  significantChange?: boolean;
}

export interface PatientValidationResult {
  valid: boolean;
  complete: boolean;
  issues: string[];
  missingFields: string[];
  bmi?: number;
  bmiCategory?: 'underweight' | 'normal' | 'overweight' | 'obese';
  warnings: string[];
  criticalAlerts: string[];
  specialPopulation: string[];
  notes: string[];
}

export interface RiskAssessment {
  score: number;
  category: string;
  tenYearRisk?: number;
  strokeRisk?: 'low' | 'moderate' | 'high';
  bleedingRisk?: 'low' | 'moderate' | 'high';
  mortality30Day?: number;
  recommendation: string[];
  error?: string;
}

export interface DrugInteractionResult {
  hasInteractions: boolean;
  severe: Array<{
    drugs: string[];
    interaction: string;
    severity: 'mild' | 'moderate' | 'severe';
    recommendation: string;
  }>;
  qtProlongation: boolean;
  qtRisk: 'low' | 'moderate' | 'high';
  recommendations: string[];
  renalAdjustmentNeeded: boolean;
  dosageAdjustments: string[];
}

export interface GuidelineValidationResult {
  compliant: boolean;
  gaps: string[];
  timeDependent: boolean;
  recommendations: string[];
}

// Validação de sinal ECG
export function validateECGSignal(
  signal: number[],
  samplingRate: number
): ECGValidationResult {
  const issues: string[] = [];
  let quality = 1.0;

  // Validar taxa de amostragem
  if (samplingRate <= 0) {
    issues.push('Invalid sampling rate');
    quality = 0;
  }

  // Validar comprimento mínimo (pelo menos 10 segundos)
  const minLength = samplingRate * 10;
  if (signal.length < minLength) {
    issues.push('Signal too short');
    quality *= 0.3;
  }

  // Detectar valores inválidos
  const hasInvalidValues = signal.some(value => 
    !isFinite(value) || isNaN(value)
  );
  if (hasInvalidValues) {
    issues.push('Invalid values detected');
    quality = 0;
  }

  // Calcular SNR (Signal-to-Noise Ratio)
  const mean = signal.reduce((sum, val) => sum + val, 0) / signal.length;
  const variance = signal.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signal.length;
  const snr = variance > 0 ? 20 * Math.log10(Math.abs(mean) / Math.sqrt(variance)) : 0;

  // Detectar ruído excessivo
  if (snr < 10) {
    issues.push('High noise level');
    quality *= 0.2;
  }

  // Detectar perda de sinal (sequências longas de zeros)
  let zeroCount = 0;
  let maxZeroSequence = 0;
  for (const value of signal) {
    if (Math.abs(value) < 0.001) {
      zeroCount++;
      maxZeroSequence = Math.max(maxZeroSequence, zeroCount);
    } else {
      zeroCount = 0;
    }
  }

  if (maxZeroSequence > samplingRate * 2) { // Mais de 2 segundos de zeros
    issues.push('Signal loss detected');
    quality *= 0.7;
  }

  // Detectar saturação (clipping)
  const maxValue = Math.max(...signal.map(Math.abs));
  const clippedValues = signal.filter(val => Math.abs(val) >= maxValue * 0.99).length;
  if (clippedValues > signal.length * 0.01) { // Mais de 1% saturado
    issues.push('Signal saturation detected');
    quality *= 0.8;
  }

  // Calcular completeness
  const completeness = 1 - (maxZeroSequence / signal.length);

  // Calcular baseline wander
  const windowSize = Math.floor(samplingRate * 0.2); // 200ms
  let baselineWander = 0;
  for (let i = windowSize; i < signal.length - windowSize; i++) {
    const window = signal.slice(i - windowSize, i + windowSize);
    const windowMean = window.reduce((sum, val) => sum + val, 0) / window.length;
    baselineWander += Math.abs(signal[i] - windowMean);
  }
  baselineWander /= (signal.length - 2 * windowSize);

  return {
    valid: quality > 0.5 && issues.length === 0,
    quality,
    issues,
    metrics: {
      snr: Math.max(0, snr),
      baselineWander,
      completeness
    }
  };
}

// Validação de frequência cardíaca
export function validateHeartRate(
  heartRate: number,
  age: number,
  _sex: 'M' | 'F',
  options?: { athlete?: boolean }
): HeartRateValidationResult {
  const warnings: string[] = [];
  const alerts: string[] = [];
  const errors: string[] = [];
  const notes: string[] = [];

  // Verificar valores impossíveis
  if (heartRate < 20 || heartRate > 250) {
    errors.push('Physiologically impossible heart rate');
    return {
      valid: false,
      category: 'normal',
      warnings,
      alerts,
      errors,
      ageAdjusted: false,
      notes
    };
  }

  // Ajustar limites por idade
  let normalMin = 60;
  let normalMax = 100;
  let ageAdjusted = false;

  if (age < 18) {
    // Pediatric ranges
    if (age < 1) {
      normalMin = 100;
      normalMax = 160;
    } else if (age < 3) {
      normalMin = 90;
      normalMax = 150;
    } else if (age < 6) {
      normalMin = 80;
      normalMax = 120;
    } else if (age < 12) {
      normalMin = 70;
      normalMax = 110;
    } else {
      normalMin = 60;
      normalMax = 105;
    }
    ageAdjusted = true;
    notes.push('Pediatric patient - adjust protocols');
  } else if (age > 65) {
    // Elderly adjustments
    normalMin = 55;
    normalMax = 95;
    ageAdjusted = true;
  }

  // Considerar atletas
  if (options?.athlete && heartRate >= 40 && heartRate < normalMin) {
    notes.push('Athletic heart rate');
    return {
      valid: true,
      category: 'athletic_bradycardia',
      warnings,
      alerts,
      errors,
      ageAdjusted,
      notes
    };
  }

  // Categorizar frequência cardíaca
  let category: HeartRateValidationResult['category'] = 'normal';
  let valid = true;

  if (heartRate < 40) {
    category = 'severe_bradycardia';
    alerts.push('Critically low heart rate');
    valid = false;
  } else if (heartRate < normalMin) {
    category = 'bradycardia';
    warnings.push('Heart rate below normal range');
  } else if (heartRate > 150) {
    category = 'severe_tachycardia';
    alerts.push('Dangerously high heart rate');
    valid = false;
  } else if (heartRate > normalMax) {
    category = 'tachycardia';
    warnings.push('Heart rate above normal range');
  }

  return {
    valid,
    category,
    warnings,
    alerts,
    errors,
    ageAdjusted,
    notes
  };
}

// Validação de pressão arterial
export function validateBloodPressure(
  systolic: number,
  diastolic: number,
  _age: number,
  _sex: 'M' | 'F'
): BloodPressureValidationResult {
  const warnings: string[] = [];
  const alerts: string[] = [];
  const errors: string[] = [];

  // Validar valores básicos
  if (diastolic >= systolic) {
    errors.push('Invalid blood pressure values');
    return {
      valid: false,
      category: 'normal',
      warnings,
      alerts,
      errors,
      map: 0,
      pulsePressure: 0,
      pulsePressureStatus: 'normal'
    };
  }

  // Calcular MAP (Mean Arterial Pressure)
  const map = diastolic + (systolic - diastolic) / 3;

  // Calcular pressão de pulso
  const pulsePressure = systolic - diastolic;
  let pulsePressureStatus: 'normal' | 'narrow' | 'wide' = 'normal';

  if (pulsePressure < 30) {
    pulsePressureStatus = 'narrow';
    warnings.push('Narrow pulse pressure');
  } else if (pulsePressure > 60) {
    pulsePressureStatus = 'wide';
    warnings.push('Wide pulse pressure');
  }

  // Categorizar pressão arterial
  let category: BloodPressureValidationResult['category'] = 'normal';
  let stage: BloodPressureValidationResult['stage'] = 'optimal';
  let valid = true;
  let urgent = false;

  if (systolic >= 180 || diastolic >= 120) {
    category = 'hypertensive_crisis';
    alerts.push('EMERGENCY: Hypertensive crisis');
    valid = false;
    urgent = true;
  } else if (systolic >= 160 || diastolic >= 100) {
    category = 'hypertension';
    stage = 'stage_2';
    alerts.push('Severe hypertension');
    valid = false;
  } else if (systolic >= 140 || diastolic >= 90) {
    category = 'hypertension';
    stage = 'stage_1';
  } else if (systolic >= 130 || diastolic >= 80) {
    category = 'elevated';
    warnings.push('Pre-hypertension');
  } else if (systolic < 90 || diastolic < 60) {
    category = 'hypotension';
    alerts.push('Low blood pressure');
    valid = false;
  }

  return {
    valid,
    category,
    stage,
    warnings,
    alerts,
    errors,
    urgent,
    map,
    pulsePressure,
    pulsePressureStatus
  };
}

// Validação de intervalo QT
export function validateQTInterval(
  qtInterval: number,
  heartRate: number,
  sex: 'M' | 'F',
  _age: number,
  options?: {
    formula?: 'Bazett' | 'Fridericia' | 'Framingham' | 'Hodges';
    medications?: string[];
    baselineQTc?: number;
  }
): QTValidationResult {
  const alerts: string[] = [];
  const warnings: string[] = [];
  const recommendations: string[] = [];

  const formula = options?.formula || 'Bazett';
  const rrInterval = 60000 / heartRate; // em ms

  // Calcular QTc baseado na fórmula
  let qtc: number;
  switch (formula) {
    case 'Fridericia':
      qtc = qtInterval / Math.pow(rrInterval / 1000, 1/3);
      break;
    case 'Framingham':
      qtc = qtInterval + 154 * (1 - rrInterval / 1000);
      break;
    case 'Hodges':
      qtc = qtInterval + 1.75 * (heartRate - 60);
      break;
    default: // Bazett
      qtc = qtInterval / Math.sqrt(rrInterval / 1000);
  }

  // Limites por sexo
  const normalLimit = sex === 'M' ? 440 : 460;
  const prolongedLimit = sex === 'M' ? 450 : 470;
  const severeLimit = sex === 'M' ? 500 : 500;

  // Categorizar QTc
  let category: QTValidationResult['category'] = 'normal';
  let risk: QTValidationResult['risk'] = 'low';
  let valid = true;

  if (qtc < 340) {
    category = 'short';
    warnings.push('Short QT syndrome suspected');
    valid = false;
  } else if (qtc > severeLimit) {
    category = 'severely_prolonged';
    risk = 'critical';
    alerts.push('High risk of Torsades de Pointes');
    valid = false;
  } else if (qtc > prolongedLimit) {
    category = 'prolonged';
    risk = 'high';
    valid = false;
  } else if (qtc > normalLimit) {
    category = 'borderline';
    risk = 'moderate';
  }

  // Verificar medicações que prolongam QT
  const qtProlongingDrugs = ['amiodarone', 'sotalol', 'quinidine', 'procainamide', 'ciprofloxacin'];
  const drugInducedRisk = options?.medications?.some(med => 
    qtProlongingDrugs.includes(med.toLowerCase())
  ) || false;

  if (drugInducedRisk) {
    recommendations.push('Monitor QT closely');
    recommendations.push('Consider alternative medications');
  }

  // Calcular delta QTc se baseline disponível
  let deltaQTc: number | undefined;
  let significantChange = false;
  if (options?.baselineQTc) {
    deltaQTc = qtc - options.baselineQTc;
    significantChange = Math.abs(deltaQTc) > 30;
  }

  return {
    valid,
    qtc,
    category,
    risk,
    alerts,
    warnings,
    formula,
    drugInducedRisk,
    recommendations,
    deltaQTc,
    significantChange
  };
}

// Validação de dados do paciente
export function validatePatientData(patient: any): PatientValidationResult {
  const issues: string[] = [];
  const missingFields: string[] = [];
  const warnings: string[] = [];
  const criticalAlerts: string[] = [];
  const specialPopulation: string[] = [];
  const notes: string[] = [];

  // Campos obrigatórios
  const requiredFields = ['id', 'name', 'age', 'sex'];
  for (const field of requiredFields) {
    if (!patient[field]) {
      missingFields.push(field);
    }
  }

  // Validar idade
  if (patient.age !== undefined) {
    if (patient.age < 0 || patient.age > 150) {
      issues.push('Invalid age');
    } else if (patient.age < 18) {
      specialPopulation.push('pediatric');
      notes.push('Pediatric patient - adjust protocols');
    } else if (patient.age > 65) {
      specialPopulation.push('geriatric');
    }
  }

  // Calcular e validar IMC
  let bmi: number | undefined;
  let bmiCategory: PatientValidationResult['bmiCategory'];

  if (patient.height && patient.weight) {
    const heightM = patient.height / 100;
    bmi = patient.weight / (heightM * heightM);

    if (bmi < 18.5) {
      bmiCategory = 'underweight';
    } else if (bmi < 25) {
      bmiCategory = 'normal';
    } else if (bmi < 30) {
      bmiCategory = 'overweight';
    } else {
      bmiCategory = 'obese';
      warnings.push('High BMI - cardiovascular risk');
    }

    // Verificar consistência altura/peso/idade
    if (patient.age < 18) {
      // Verificações pediátricas simplificadas
      if (patient.age < 5 && (patient.height > 120 || patient.weight > 25)) {
        issues.push('Inconsistent height/weight for age');
      }
    }
  }

  // Verificar alergias vs medicações
  if (patient.allergies && patient.medications) {
    const allergyConflicts = patient.medications.filter((med: string) => {
      return patient.allergies.some((allergy: string) => {
        // Verificações simplificadas de conflitos
        if (allergy.toLowerCase().includes('penicillin') && 
            med.toLowerCase().includes('amoxicillin')) {
          return true;
        }
        return allergy.toLowerCase() === med.toLowerCase();
      });
    });

    if (allergyConflicts.length > 0) {
      criticalAlerts.push('Medication allergy conflict');
    }
  }

  const valid = missingFields.length === 0 && issues.length === 0 && criticalAlerts.length === 0;
  const complete = missingFields.length === 0;

  return {
    valid,
    complete,
    issues,
    missingFields,
    bmi,
    bmiCategory,
    warnings,
    criticalAlerts,
    specialPopulation,
    notes
  };
}

// Cálculo de scores de risco
export function calculateRiskScore(
  patient: any,
  scoreType: 'SCORE2' | 'CHA2DS2-VASc' | 'HAS-BLED' | 'TIMI'
): RiskAssessment {
  switch (scoreType) {
    case 'SCORE2':
      return calculateSCORE2(patient);
    case 'CHA2DS2-VASc':
      return calculateCHA2DS2VASc(patient);
    case 'HAS-BLED':
      return calculateHASBLED(patient);
    case 'TIMI':
      return calculateTIMI(patient);
    default:
      return {
        score: 0,
        category: 'unknown',
        recommendation: [],
        error: 'Unknown risk score type'
      };
  }
}

function calculateSCORE2(patient: any): RiskAssessment {
  let score = 0;
  
  // Idade
  if (patient.age >= 70) score += 4;
  else if (patient.age >= 65) score += 3;
  else if (patient.age >= 60) score += 2;
  else if (patient.age >= 55) score += 1;

  // Sexo
  if (patient.sex === 'M') score += 1;

  // Tabagismo
  if (patient.vitals?.smoking) score += 2;

  // Pressão arterial sistólica
  const systolic = patient.vitals?.bloodPressure?.systolic || 120;
  if (systolic >= 180) score += 4;
  else if (systolic >= 160) score += 3;
  else if (systolic >= 140) score += 2;
  else if (systolic >= 120) score += 1;

  // Colesterol
  const cholesterol = patient.labs?.cholesterol || 200;
  if (cholesterol >= 310) score += 3;
  else if (cholesterol >= 270) score += 2;
  else if (cholesterol >= 230) score += 1;

  const tenYearRisk = Math.min(score * 2, 40); // Simplificado

  return {
    score,
    category: tenYearRisk < 5 ? 'low' : tenYearRisk < 10 ? 'moderate' : 'high',
    tenYearRisk,
    recommendation: tenYearRisk > 10 ? ['Consider statin therapy', 'Lifestyle modifications'] : ['Lifestyle modifications']
  };
}

function calculateCHA2DS2VASc(patient: any): RiskAssessment {
  let score = 0;

  // C - CHF
  if (patient.conditions?.includes('heart_failure')) score += 1;
  
  // H - Hypertension
  if (patient.conditions?.includes('hypertension')) score += 1;
  
  // A2 - Age >= 75
  if (patient.age >= 75) score += 2;
  else if (patient.age >= 65) score += 1; // A - Age 65-74
  
  // D - Diabetes
  if (patient.conditions?.includes('diabetes')) score += 1;
  
  // S2 - Stroke/TIA history
  if (patient.conditions?.includes('stroke') || patient.conditions?.includes('tia')) score += 2;
  
  // V - Vascular disease
  if (patient.conditions?.includes('vascular_disease')) score += 1;
  
  // Sc - Sex category (female)
  if (patient.sex === 'F') score += 1;

  const strokeRisk = score === 0 ? 'low' : score === 1 ? 'moderate' : 'high';

  return {
    score,
    category: strokeRisk,
    strokeRisk: strokeRisk as 'low' | 'moderate' | 'high',
    recommendation: score >= 2 ? ['anticoagulation'] : score === 1 ? ['Consider anticoagulation'] : ['No anticoagulation needed']
  };
}

function calculateHASBLED(patient: any): RiskAssessment {
  let score = 0;

  // H - Hypertension
  if (patient.conditions?.includes('hypertension')) score += 1;
  
  // A - Abnormal renal/liver function
  if (patient.conditions?.includes('renal_disease') || patient.conditions?.includes('liver_disease')) score += 1;
  
  // S - Stroke
  if (patient.conditions?.includes('stroke')) score += 1;
  
  // B - Bleeding history
  if (patient.conditions?.includes('bleeding_history')) score += 1;
  
  // L - Labile INR
  if (patient.labs?.inr && patient.labs.inr > 3.0) score += 1;
  
  // E - Elderly (>65)
  if (patient.age > 65) score += 1;
  
  // D - Drugs/alcohol
  if (patient.medications?.includes('warfarin') || patient.medications?.includes('aspirin')) score += 1;

  const bleedingRisk = score < 3 ? 'low' : score < 5 ? 'moderate' : 'high';

  return {
    score,
    category: bleedingRisk,
    bleedingRisk: bleedingRisk as 'low' | 'moderate' | 'high',
    recommendation: score >= 3 ? ['Caution with anticoagulation', 'Regular monitoring'] : ['Standard anticoagulation protocols']
  };
}

function calculateTIMI(patient: any): RiskAssessment {
  let score = 0;

  // Age >= 65
  if (patient.age >= 65) score += 1;
  
  // >= 3 CAD risk factors
  let riskFactors = 0;
  if (patient.conditions?.includes('hypertension')) riskFactors++;
  if (patient.conditions?.includes('diabetes')) riskFactors++;
  if (patient.vitals?.smoking) riskFactors++;
  if (patient.labs?.cholesterol > 240) riskFactors++;
  if (riskFactors >= 3) score += 1;
  
  // Known CAD
  if (patient.conditions?.includes('cad')) score += 1;
  
  // Aspirin use in past 7 days
  if (patient.medications?.includes('aspirin')) score += 1;
  
  // Severe anginal symptoms
  if (patient.conditions?.includes('unstable_angina')) score += 1;
  
  // ST changes
  if (patient.ecgChanges) score += 1;
  
  // Elevated cardiac markers
  if (patient.troponin && patient.troponin > 0.1) score += 1;

  const mortality30Day = score * 2.5; // Simplificado

  return {
    score,
    category: score < 3 ? 'low' : score < 5 ? 'intermediate' : 'high',
    mortality30Day,
    recommendation: score >= 5 ? ['Urgent invasive strategy'] : ['Conservative management']
  };
}

// Validação de medicações
export function validateMedication(
  medication: string,
  patient: any
): { valid: boolean; warnings: string[]; contraindications: string[] } {
  const warnings: string[] = [];
  const contraindications: string[] = [];

  // Verificações básicas de alergias
  if (patient.allergies?.includes(medication.toLowerCase())) {
    contraindications.push('Patient allergic to this medication');
  }

  // Verificações específicas por medicação
  switch (medication.toLowerCase()) {
    case 'metformin':
      if (patient.labs?.gfr && patient.labs.gfr < 30) {
        contraindications.push('Contraindicated in severe renal impairment');
      }
      break;
    case 'warfarin':
      if (patient.conditions?.includes('bleeding_history')) {
        warnings.push('Caution: history of bleeding');
      }
      break;
  }

  return {
    valid: contraindications.length === 0,
    warnings,
    contraindications
  };
}

// Verificação de interações medicamentosas
export function checkDrugInteractions(
  medications: string[],
  options?: { renalFunction?: { gfr: number } }
): DrugInteractionResult {
  const severe: DrugInteractionResult['severe'] = [];
  const recommendations: string[] = [];
  const dosageAdjustments: string[] = [];

  // Interações conhecidas
  const interactions = [
    {
      drugs: ['warfarin', 'aspirin'],
      interaction: 'Increased bleeding risk',
      severity: 'severe' as const,
      recommendation: 'Monitor INR closely, consider PPI'
    },
    {
      drugs: ['amiodarone', 'digoxin'],
      interaction: 'Increased digoxin levels',
      severity: 'moderate' as const,
      recommendation: 'Reduce digoxin dose by 50%'
    }
  ];

  // Verificar interações
  for (const interaction of interactions) {
    const hasAllDrugs = interaction.drugs.every(drug => 
      medications.some(med => med.toLowerCase().includes(drug.toLowerCase()))
    );
    
    if (hasAllDrugs) {
      severe.push(interaction);
    }
  }

  // Medicações que prolongam QT
  const qtProlongingDrugs = ['amiodarone', 'sotalol', 'ciprofloxacin'];
  const qtMedications = medications.filter(med => 
    qtProlongingDrugs.some(qtDrug => med.toLowerCase().includes(qtDrug.toLowerCase()))
  );

  const qtProlongation = qtMedications.length > 0;
  const qtRisk = qtMedications.length > 1 ? 'high' : qtMedications.length === 1 ? 'moderate' : 'low';

  if (qtProlongation) {
    recommendations.push('Monitor QT interval closely');
  }

  // Ajustes renais
  const renalAdjustmentDrugs = ['metformin', 'digoxin', 'atenolol'];
  let renalAdjustmentNeeded = false;

  if (options?.renalFunction?.gfr && options.renalFunction.gfr < 60) {
    for (const med of medications) {
      if (renalAdjustmentDrugs.some(drug => med.toLowerCase().includes(drug.toLowerCase()))) {
        dosageAdjustments.push(med);
        renalAdjustmentNeeded = true;
      }
    }
  }

  return {
    hasInteractions: severe.length > 0,
    severe,
    qtProlongation,
    qtRisk: qtRisk as 'low' | 'moderate' | 'high',
    recommendations,
    renalAdjustmentNeeded,
    dosageAdjustments
  };
}

// Validação de consistência diagnóstica
export function validateDiagnosisConsistency(
  diagnosis: any,
  ecgFindings: any,
  _clinicalData: any
): { consistent: boolean; conflicts: string[]; recommendations: string[] } {
  const conflicts: string[] = [];
  const recommendations: string[] = [];

  // Verificações básicas de consistência
  if (diagnosis.condition === 'STEMI' && !ecgFindings.stElevation) {
    conflicts.push('STEMI diagnosis without ST elevation');
  }

  if (diagnosis.condition === 'atrial_fibrillation' && ecgFindings.rhythm === 'sinus') {
    conflicts.push('AF diagnosis with sinus rhythm');
  }

  // Recomendações baseadas em achados
  if (ecgFindings.qtProlonged && !diagnosis.conditions?.includes('qt_prolongation')) {
    recommendations.push('Consider QT prolongation in differential');
  }

  return {
    consistent: conflicts.length === 0,
    conflicts,
    recommendations
  };
}

// Validação de diretrizes clínicas
export function validateClinicalGuidelines(
  diagnosis: any,
  _guidelineVersion: string
): GuidelineValidationResult {
  const gaps: string[] = [];
  const recommendations: string[] = [];
  let timeDependent = false;

  switch (diagnosis.condition) {
    case 'atrial_fibrillation':
      // Verificar anticoagulação
      if (diagnosis.patient?.age > 65 && !diagnosis.treatment?.anticoagulation) {
        gaps.push('Anticoagulation indicated but not prescribed');
      }
      break;

    case 'STEMI':
      timeDependent = true;
      if (diagnosis.timeFromOnset < 12 && !diagnosis.treatment?.pci && !diagnosis.treatment?.thrombolysis) {
        gaps.push('Reperfusion therapy not initiated');
      }
      break;
  }

  return {
    compliant: gaps.length === 0,
    gaps,
    timeDependent,
    recommendations
  };
}

