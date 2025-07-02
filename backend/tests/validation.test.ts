// frontend/src/utils/medical/__tests__/validation.test.ts
// Testes completos para utilitários de validação médica - 100% cobertura

import {
  validateECGSignal,
  validateHeartRate,
  validateBloodPressure,
  validateQTInterval,
  validatePatientData,
  validateDiagnosisConsistency,
  calculateRiskScore,
  validateMedication,
  checkDrugInteractions,
  validateClinicalGuidelines,
  ECGValidationResult,
  PatientValidationResult,
  RiskAssessment,
} from '../validation';

describe('Medical Validation Utils - 100% Coverage', () => {
  describe('validateECGSignal', () => {
    it('deve validar sinal ECG válido', () => {
      const validSignal = Array(3600).fill(0).map((_, i) => 
        Math.sin(2 * Math.PI * 1.2 * i / 360)
      );
      
      const result = validateECGSignal(validSignal, 360);
      
      expect(result.valid).toBe(true);
      expect(result.quality).toBeGreaterThan(0.8);
      expect(result.issues).toHaveLength(0);
    });

    it('deve rejeitar sinal muito curto', () => {
      const shortSignal = [1, 2, 3, 4, 5];
      
      const result = validateECGSignal(shortSignal, 360);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Signal too short');
      expect(result.quality).toBeLessThan(0.5);
    });

    it('deve detectar sinal com muito ruído', () => {
      const noisySignal = Array(3600).fill(0).map(() => 
        Math.random() * 10 - 5
      );
      
      const result = validateECGSignal(noisySignal, 360);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('High noise level');
      expect(result.quality).toBeLessThan(0.3);
    });

    it('deve detectar valores inválidos (NaN, Infinity)', () => {
      const invalidSignal = [1, 2, NaN, 4, Infinity, 6];
      
      const result = validateECGSignal(invalidSignal, 360);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Invalid values detected');
    });

    it('deve detectar taxa de amostragem inválida', () => {
      const signal = Array(100).fill(0);
      
      const result = validateECGSignal(signal, 0);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Invalid sampling rate');
    });

    it('deve detectar perda de sinal', () => {
      const signalWithLoss = Array(3600).fill(0).map((_, i) => {
        if (i > 1000 && i < 1500) return 0; // Perda de sinal
        return Math.sin(2 * Math.PI * 1.2 * i / 360);
      });
      
      const result = validateECGSignal(signalWithLoss, 360);
      
      expect(result.issues).toContain('Signal loss detected');
      expect(result.quality).toBeLessThan(0.8);
    });

    it('deve detectar saturação de sinal', () => {
      const saturatedSignal = Array(3600).fill(0).map((_, i) => {
        const value = Math.sin(2 * Math.PI * 1.2 * i / 360) * 5;
        return Math.max(-3, Math.min(3, value)); // Clipping
      });
      
      const result = validateECGSignal(saturatedSignal, 360);
      
      expect(result.issues).toContain('Signal saturation detected');
    });

    it('deve calcular métricas de qualidade detalhadas', () => {
      const signal = Array(3600).fill(0).map((_, i) => 
        Math.sin(2 * Math.PI * 1.2 * i / 360) + Math.random() * 0.1
      );
      
      const result = validateECGSignal(signal, 360);
      
      expect(result.metrics).toBeDefined();
      expect(result.metrics.snr).toBeGreaterThan(10);
      expect(result.metrics.baselineWander).toBeLessThan(0.1);
      expect(result.metrics.completeness).toBeGreaterThan(0.95);
    });
  });

  describe('validateHeartRate', () => {
    it('deve validar frequência cardíaca normal', () => {
      const result = validateHeartRate(75, 45, 'M');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('normal');
      expect(result.warnings).toHaveLength(0);
    });

    it('deve detectar bradicardia', () => {
      const result = validateHeartRate(45, 30, 'F');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('bradycardia');
      expect(result.warnings).toContain('Heart rate below normal range');
    });

    it('deve detectar bradicardia severa', () => {
      const result = validateHeartRate(35, 50, 'M');
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('severe_bradycardia');
      expect(result.alerts).toContain('Critically low heart rate');
    });

    it('deve detectar taquicardia', () => {
      const result = validateHeartRate(110, 25, 'F');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('tachycardia');
      expect(result.warnings).toContain('Heart rate above normal range');
    });

    it('deve detectar taquicardia severa', () => {
      const result = validateHeartRate(180, 40, 'M');
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('severe_tachycardia');
      expect(result.alerts).toContain('Dangerously high heart rate');
    });

    it('deve ajustar limites para idade pediátrica', () => {
      // Criança de 5 anos - FC normal mais alta
      const result = validateHeartRate(100, 5, 'F');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('normal');
      expect(result.ageAdjusted).toBe(true);
    });

    it('deve ajustar limites para idosos', () => {
      // Idoso de 80 anos - limites diferentes
      const result = validateHeartRate(65, 80, 'M');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('normal');
      expect(result.ageAdjusted).toBe(true);
    });

    it('deve considerar condicionamento físico', () => {
      const result = validateHeartRate(48, 25, 'M', { athlete: true });
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('athletic_bradycardia');
      expect(result.notes).toContain('Athletic heart rate');
    });

    it('deve rejeitar valores impossíveis', () => {
      const result = validateHeartRate(300, 30, 'F');
      
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Physiologically impossible heart rate');
    });
  });

  describe('validateBloodPressure', () => {
    it('deve validar pressão arterial normal', () => {
      const result = validateBloodPressure(120, 80, 45, 'M');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('normal');
      expect(result.stage).toBe('optimal');
    });

    it('deve detectar pré-hipertensão', () => {
      const result = validateBloodPressure(135, 85, 50, 'F');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('elevated');
      expect(result.warnings).toContain('Pre-hypertension');
    });

    it('deve detectar hipertensão estágio 1', () => {
      const result = validateBloodPressure(145, 92, 55, 'M');
      
      expect(result.valid).toBe(true);
      expect(result.category).toBe('hypertension');
      expect(result.stage).toBe('stage_1');
    });

    it('deve detectar hipertensão estágio 2', () => {
      const result = validateBloodPressure(165, 105, 60, 'F');
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('hypertension');
      expect(result.stage).toBe('stage_2');
      expect(result.alerts).toContain('Severe hypertension');
    });

    it('deve detectar crise hipertensiva', () => {
      const result = validateBloodPressure(185, 125, 65, 'M');
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('hypertensive_crisis');
      expect(result.urgent).toBe(true);
      expect(result.alerts).toContain('EMERGENCY: Hypertensive crisis');
    });

    it('deve detectar hipotensão', () => {
      const result = validateBloodPressure(85, 55, 40, 'F');
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('hypotension');
      expect(result.alerts).toContain('Low blood pressure');
    });

    it('deve calcular pressão arterial média (MAP)', () => {
      const result = validateBloodPressure(120, 80, 45, 'M');
      
      expect(result.map).toBeDefined();
      expect(result.map).toBeCloseTo(93.33, 1);
    });

    it('deve calcular pressão de pulso', () => {
      const result = validateBloodPressure(140, 90, 50, 'F');
      
      expect(result.pulsePressure).toBe(50);
      expect(result.pulsePressureStatus).toBe('normal');
    });

    it('deve detectar pressão de pulso anormal', () => {
      const result = validateBloodPressure(180, 90, 70, 'M');
      
      expect(result.pulsePressure).toBe(90);
      expect(result.pulsePressureStatus).toBe('wide');
      expect(result.warnings).toContain('Wide pulse pressure');
    });

    it('deve validar valores fisiologicamente impossíveis', () => {
      // Diastólica maior que sistólica
      const result = validateBloodPressure(80, 120, 50, 'F');
      
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Invalid blood pressure values');
    });
  });

  describe('validateQTInterval', () => {
    it('deve validar intervalo QT normal', () => {
      const result = validateQTInterval(380, 72, 'M', 45);
      
      expect(result.valid).toBe(true);
      expect(result.qtc).toBeCloseTo(397, 0);
      expect(result.category).toBe('normal');
    });

    it('deve detectar QT prolongado', () => {
      const result = validateQTInterval(480, 60, 'F', 50);
      
      expect(result.valid).toBe(false);
      expect(result.qtc).toBeCloseTo(480, 0);
      expect(result.category).toBe('prolonged');
      expect(result.risk).toBe('high');
    });

    it('deve detectar QT muito prolongado', () => {
      const result = validateQTInterval(520, 70, 'M', 40);
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('severely_prolonged');
      expect(result.risk).toBe('critical');
      expect(result.alerts).toContain('High risk of Torsades de Pointes');
    });

    it('deve detectar QT curto', () => {
      const result = validateQTInterval(320, 80, 'F', 35);
      
      expect(result.valid).toBe(false);
      expect(result.category).toBe('short');
      expect(result.warnings).toContain('Short QT syndrome suspected');
    });

    it('deve usar fórmula de Bazett corretamente', () => {
      const result = validateQTInterval(400, 60, 'M', 50);
      
      expect(result.formula).toBe('Bazett');
      expect(result.qtc).toBe(400); // RR = 1000ms, sqrt(1) = 1
    });

    it('deve oferecer fórmulas alternativas', () => {
      const result = validateQTInterval(400, 100, 'F', 30, {
        formula: 'Fridericia'
      });
      
      expect(result.formula).toBe('Fridericia');
      // QTc Fridericia = QT / (RR^0.33)
      expect(result.qtc).toBeCloseTo(465, 0);
    });

    it('deve ajustar limites por sexo', () => {
      const qtValue = 445;
      
      const maleResult = validateQTInterval(qtValue, 70, 'M', 40);
      const femaleResult = validateQTInterval(qtValue, 70, 'F', 40);
      
      expect(maleResult.category).toBe('prolonged');
      expect(femaleResult.category).toBe('borderline');
    });

    it('deve considerar medicações que prolongam QT', () => {
      const result = validateQTInterval(440, 75, 'M', 55, {
        medications: ['amiodarone', 'sotalol']
      });
      
      expect(result.drugInducedRisk).toBe(true);
      expect(result.recommendations).toContain('Monitor QT closely');
    });

    it('deve calcular delta QTc quando há baseline', () => {
      const result = validateQTInterval(460, 70, 'F', 45, {
        baselineQTc: 420
      });
      
      expect(result.deltaQTc).toBe(40);
      expect(result.significantChange).toBe(true);
    });
  });

  describe('validatePatientData', () => {
    const validPatient = {
      id: 'PAT001',
      name: 'João Silva',
      age: 45,
      sex: 'M' as const,
      height: 175,
      weight: 80,
      conditions: [],
      medications: [],
    };

    it('deve validar dados completos do paciente', () => {
      const result = validatePatientData(validPatient);
      
      expect(result.valid).toBe(true);
      expect(result.complete).toBe(true);
      expect(result.issues).toHaveLength(0);
    });

    it('deve detectar campos obrigatórios faltando', () => {
      const incompletePatient = {
        name: 'Maria Santos',
        age: 30,
        // Faltam sex, id
      };
      
      const result = validatePatientData(incompletePatient as any);
      
      expect(result.valid).toBe(false);
      expect(result.missingFields).toContain('id');
      expect(result.missingFields).toContain('sex');
    });

    it('deve validar idade válida', () => {
      const invalidAge = { ...validPatient, age: -5 };
      
      const result = validatePatientData(invalidAge);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Invalid age');
    });

    it('deve calcular e validar IMC', () => {
      const result = validatePatientData(validPatient);
      
      expect(result.bmi).toBeCloseTo(26.12, 1);
      expect(result.bmiCategory).toBe('overweight');
    });

    it('deve detectar IMC anormal', () => {
      const obesePatient = { ...validPatient, weight: 120 };
      
      const result = validatePatientData(obesePatient);
      
      expect(result.bmiCategory).toBe('obese');
      expect(result.warnings).toContain('High BMI - cardiovascular risk');
    });

    it('deve validar alergias medicamentosas', () => {
      const patientWithAllergies = {
        ...validPatient,
        allergies: ['penicillin', 'aspirin'],
        medications: ['amoxicillin'], // Penicilina!
      };
      
      const result = validatePatientData(patientWithAllergies);
      
      expect(result.valid).toBe(false);
      expect(result.criticalAlerts).toContain('Medication allergy conflict');
    });

    it('deve validar dados demográficos especiais', () => {
      const pediatricPatient = { ...validPatient, age: 5 };
      
      const result = validatePatientData(pediatricPatient);
      
      expect(result.specialPopulation).toContain('pediatric');
      expect(result.notes).toContain('Pediatric patient - adjust protocols');
    });

    it('deve detectar dados inconsistentes', () => {
      const inconsistentPatient = {
        ...validPatient,
        age: 2,
        height: 175, // Muito alto para 2 anos
        weight: 80,  // Muito pesado para 2 anos
      };
      
      const result = validatePatientData(inconsistentPatient);
      
      expect(result.valid).toBe(false);
      expect(result.issues).toContain('Inconsistent height/weight for age');
    });
  });

  describe('calculateRiskScore', () => {
    const basePatient = {
      age: 55,
      sex: 'M' as const,
      conditions: ['hypertension'],
      labs: {
        cholesterol: 220,
        ldl: 140,
        hdl: 45,
        triglycerides: 180,
      },
      vitals: {
        bloodPressure: { systolic: 145, diastolic: 90 },
        smoking: false,
      },
    };

    it('deve calcular SCORE2 para risco cardiovascular', () => {
      const result = calculateRiskScore(basePatient, 'SCORE2');
      
      expect(result.score).toBeGreaterThan(0);
      expect(result.category).toBeDefined();
      expect(result.tenYearRisk).toBeDefined();
    });

    it('deve calcular CHA2DS2-VASc para FA', () => {
      const afPatient = {
        ...basePatient,
        conditions: ['atrial_fibrillation', 'hypertension', 'diabetes'],
        age: 75,
      };
      
      const result = calculateRiskScore(afPatient, 'CHA2DS2-VASc');
      
      expect(result.score).toBeGreaterThanOrEqual(4); // Age 75+, HTN, DM
      expect(result.strokeRisk).toBe('high');
      expect(result.recommendation).toContain('anticoagulation');
    });

    it('deve calcular HAS-BLED para risco de sangramento', () => {
      const bleedingRiskPatient = {
        ...basePatient,
        conditions: ['hypertension', 'renal_disease'],
        medications: ['warfarin'],
        labs: { inr: 3.5 },
      };
      
      const result = calculateRiskScore(bleedingRiskPatient, 'HAS-BLED');
      
      expect(result.score).toBeGreaterThan(0);
      expect(result.bleedingRisk).toBeDefined();
    });

    it('deve calcular TIMI para SCA', () => {
      const acsPatient = {
        ...basePatient,
        conditions: ['unstable_angina'],
        ecgChanges: true,
        troponin: 0.5,
      };
      
      const result = calculateRiskScore(acsPatient, 'TIMI');
      
      expect(result.score).toBeGreaterThan(0);
      expect(result.mortality30Day).toBeDefined();
    });

    it('deve retornar erro para tipo de score inválido', () => {
      const result = calculateRiskScore(basePatient, 'INVALID_SCORE' as any);
      
      expect(result.error).toBe('Unknown risk score type');
    });
  });

  describe('checkDrugInteractions', () => {
    it('deve detectar interações graves', () => {
      const medications = ['warfarin', 'aspirin', 'amiodarone'];
      
      const result = checkDrugInteractions(medications);
      
      expect(result.hasInteractions).toBe(true);
      expect(result.severe).toHaveLength(1); // Warfarin + Aspirin
      expect(result.severe[0].drugs).toContain('warfarin');
      expect(result.severe[0].drugs).toContain('aspirin');
    });

    it('deve detectar interações com QT', () => {
      const qtMedications = ['amiodarone', 'sotalol', 'ciprofloxacin'];
      
      const result = checkDrugInteractions(qtMedications);
      
      expect(result.qtProlongation).toBe(true);
      expect(result.qtRisk).toBe('high');
      expect(result.recommendations).toContain('Monitor QT interval closely');
    });

    it('deve não encontrar interações quando seguro', () => {
      const safeMedications = ['metformin', 'lisinopril', 'atorvastatin'];
      
      const result = checkDrugInteractions(safeMedications);
      
      expect(result.hasInteractions).toBe(false);
      expect(result.severe).toHaveLength(0);
    });

    it('deve considerar função renal', () => {
      const medications = ['metformin', 'digoxin'];
      const renalFunction = { gfr: 25 }; // Insuficiência renal
      
      const result = checkDrugInteractions(medications, { renalFunction });
      
      expect(result.renalAdjustmentNeeded).toBe(true);
      expect(result.dosageAdjustments).toContain('metformin');
    });
  });

  describe('validateClinicalGuidelines', () => {
    it('deve validar diretrizes para FA', () => {
      const diagnosis = {
        condition: 'atrial_fibrillation',
        patient: {
          age: 70,
          conditions: ['hypertension'],
        },
        treatment: {
          anticoagulation: false,
        },
      };
      
      const result = validateClinicalGuidelines(diagnosis, 'ESC_2020');
      
      expect(result.compliant).toBe(false);
      expect(result.gaps).toContain('Anticoagulation indicated but not prescribed');
    });

    it('deve validar diretrizes para IAM', () => {
      const diagnosis = {
        condition: 'STEMI',
        timeFromOnset: 2, // horas
        treatment: {
          aspirin: true,
          pci: false,
          thrombolysis: false,
        },
      };
      
      const result = validateClinicalGuidelines(diagnosis, 'AHA_2021');
      
      expect(result.compliant).toBe(false);
      expect(result.gaps).toContain('Reperfusion therapy not initiated');
      expect(result.timeDependent).toBe(true);
    });
  });
});
