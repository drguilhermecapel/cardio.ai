"""
Schemas FHIR R4 para CardioAI
Implementa estruturas de dados compatíveis com FHIR R4
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class FHIRResourceType(str, Enum):
    """Tipos de recursos FHIR."""
    OBSERVATION = "Observation"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PATIENT = "Patient"
    PRACTITIONER = "Practitioner"
    ORGANIZATION = "Organization"


class ObservationStatus(str, Enum):
    """Status de observação FHIR."""
    REGISTERED = "registered"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class DiagnosticReportStatus(str, Enum):
    """Status de relatório diagnóstico FHIR."""
    REGISTERED = "registered"
    PARTIAL = "partial"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    APPENDED = "appended"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class FHIRCoding(BaseModel):
    """Codificação FHIR."""
    system: Optional[str] = Field(None, description="Sistema de codificação")
    version: Optional[str] = Field(None, description="Versão do sistema")
    code: Optional[str] = Field(None, description="Código")
    display: Optional[str] = Field(None, description="Texto de exibição")
    userSelected: Optional[bool] = Field(None, description="Selecionado pelo usuário")


class FHIRCodeableConcept(BaseModel):
    """Conceito codificável FHIR."""
    coding: Optional[List[FHIRCoding]] = Field(None, description="Codificações")
    text: Optional[str] = Field(None, description="Texto livre")


class FHIRQuantity(BaseModel):
    """Quantidade FHIR."""
    value: Optional[float] = Field(None, description="Valor numérico")
    comparator: Optional[str] = Field(None, description="Comparador (<, <=, >=, >)")
    unit: Optional[str] = Field(None, description="Unidade")
    system: Optional[str] = Field(None, description="Sistema de unidades")
    code: Optional[str] = Field(None, description="Código da unidade")


class FHIRReference(BaseModel):
    """Referência FHIR."""
    reference: Optional[str] = Field(None, description="Referência literal")
    type: Optional[str] = Field(None, description="Tipo do recurso")
    identifier: Optional[Dict[str, Any]] = Field(None, description="Identificador lógico")
    display: Optional[str] = Field(None, description="Texto de exibição")


class FHIRPeriod(BaseModel):
    """Período FHIR."""
    start: Optional[datetime] = Field(None, description="Data/hora de início")
    end: Optional[datetime] = Field(None, description="Data/hora de fim")


class FHIRObservationComponent(BaseModel):
    """Componente de observação FHIR."""
    code: FHIRCodeableConcept = Field(..., description="Tipo de componente")
    valueQuantity: Optional[FHIRQuantity] = Field(None, description="Valor quantitativo")
    valueCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="Valor codificado")
    valueString: Optional[str] = Field(None, description="Valor textual")
    valueBoolean: Optional[bool] = Field(None, description="Valor booleano")
    valueInteger: Optional[int] = Field(None, description="Valor inteiro")
    valueRange: Optional[Dict[str, Any]] = Field(None, description="Valor de intervalo")
    valueRatio: Optional[Dict[str, Any]] = Field(None, description="Valor de razão")
    valueSampledData: Optional[Dict[str, Any]] = Field(None, description="Dados amostrados")
    valueTime: Optional[str] = Field(None, description="Valor de tempo")
    valueDateTime: Optional[datetime] = Field(None, description="Valor de data/hora")
    valuePeriod: Optional[FHIRPeriod] = Field(None, description="Valor de período")
    dataAbsentReason: Optional[FHIRCodeableConcept] = Field(None, description="Razão da ausência de dados")
    interpretation: Optional[List[FHIRCodeableConcept]] = Field(None, description="Interpretação")
    referenceRange: Optional[List[Dict[str, Any]]] = Field(None, description="Intervalo de referência")


class FHIRObservation(BaseModel):
    """Observação FHIR R4."""
    resourceType: str = Field(default="Observation", description="Tipo do recurso")
    id: Optional[str] = Field(None, description="ID lógico do recurso")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadados")
    implicitRules: Optional[str] = Field(None, description="Regras implícitas")
    language: Optional[str] = Field(None, description="Idioma")
    text: Optional[Dict[str, Any]] = Field(None, description="Narrativa textual")
    contained: Optional[List[Dict[str, Any]]] = Field(None, description="Recursos contidos")
    extension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões")
    modifierExtension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões modificadoras")
    
    # Elementos específicos da Observação
    identifier: Optional[List[Dict[str, Any]]] = Field(None, description="Identificadores")
    basedOn: Optional[List[FHIRReference]] = Field(None, description="Baseado em")
    partOf: Optional[List[FHIRReference]] = Field(None, description="Parte de")
    status: ObservationStatus = Field(..., description="Status da observação")
    category: Optional[List[FHIRCodeableConcept]] = Field(None, description="Categoria")
    code: FHIRCodeableConcept = Field(..., description="Tipo de observação")
    subject: Optional[FHIRReference] = Field(None, description="Sujeito da observação")
    focus: Optional[List[FHIRReference]] = Field(None, description="Foco da observação")
    encounter: Optional[FHIRReference] = Field(None, description="Encontro")
    effectiveDateTime: Optional[datetime] = Field(None, description="Data/hora efetiva")
    effectivePeriod: Optional[FHIRPeriod] = Field(None, description="Período efetivo")
    effectiveTiming: Optional[Dict[str, Any]] = Field(None, description="Timing efetivo")
    effectiveInstant: Optional[datetime] = Field(None, description="Instante efetivo")
    issued: Optional[datetime] = Field(None, description="Data de emissão")
    performer: Optional[List[FHIRReference]] = Field(None, description="Executores")
    
    # Valores da observação
    valueQuantity: Optional[FHIRQuantity] = Field(None, description="Valor quantitativo")
    valueCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="Valor codificado")
    valueString: Optional[str] = Field(None, description="Valor textual")
    valueBoolean: Optional[bool] = Field(None, description="Valor booleano")
    valueInteger: Optional[int] = Field(None, description="Valor inteiro")
    valueRange: Optional[Dict[str, Any]] = Field(None, description="Valor de intervalo")
    valueRatio: Optional[Dict[str, Any]] = Field(None, description="Valor de razão")
    valueSampledData: Optional[Dict[str, Any]] = Field(None, description="Dados amostrados")
    valueTime: Optional[str] = Field(None, description="Valor de tempo")
    valueDateTime: Optional[datetime] = Field(None, description="Valor de data/hora")
    valuePeriod: Optional[FHIRPeriod] = Field(None, description="Valor de período")
    
    dataAbsentReason: Optional[FHIRCodeableConcept] = Field(None, description="Razão da ausência de dados")
    interpretation: Optional[List[FHIRCodeableConcept]] = Field(None, description="Interpretação")
    note: Optional[List[Dict[str, Any]]] = Field(None, description="Comentários")
    bodySite: Optional[FHIRCodeableConcept] = Field(None, description="Local do corpo")
    method: Optional[FHIRCodeableConcept] = Field(None, description="Método")
    specimen: Optional[FHIRReference] = Field(None, description="Espécime")
    device: Optional[FHIRReference] = Field(None, description="Dispositivo")
    referenceRange: Optional[List[Dict[str, Any]]] = Field(None, description="Intervalo de referência")
    hasMember: Optional[List[FHIRReference]] = Field(None, description="Tem membro")
    derivedFrom: Optional[List[FHIRReference]] = Field(None, description="Derivado de")
    component: Optional[List[FHIRObservationComponent]] = Field(None, description="Componentes")


class FHIRDiagnosticReport(BaseModel):
    """Relatório diagnóstico FHIR R4."""
    resourceType: str = Field(default="DiagnosticReport", description="Tipo do recurso")
    id: Optional[str] = Field(None, description="ID lógico do recurso")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadados")
    implicitRules: Optional[str] = Field(None, description="Regras implícitas")
    language: Optional[str] = Field(None, description="Idioma")
    text: Optional[Dict[str, Any]] = Field(None, description="Narrativa textual")
    contained: Optional[List[Dict[str, Any]]] = Field(None, description="Recursos contidos")
    extension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões")
    modifierExtension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões modificadoras")
    
    # Elementos específicos do Relatório Diagnóstico
    identifier: Optional[List[Dict[str, Any]]] = Field(None, description="Identificadores")
    basedOn: Optional[List[FHIRReference]] = Field(None, description="Baseado em")
    status: DiagnosticReportStatus = Field(..., description="Status do relatório")
    category: Optional[List[FHIRCodeableConcept]] = Field(None, description="Categoria")
    code: FHIRCodeableConcept = Field(..., description="Tipo de relatório")
    subject: Optional[FHIRReference] = Field(None, description="Sujeito do relatório")
    encounter: Optional[FHIRReference] = Field(None, description="Encontro")
    effectiveDateTime: Optional[datetime] = Field(None, description="Data/hora efetiva")
    effectivePeriod: Optional[FHIRPeriod] = Field(None, description="Período efetivo")
    issued: Optional[datetime] = Field(None, description="Data de emissão")
    performer: Optional[List[FHIRReference]] = Field(None, description="Executores")
    resultsInterpreter: Optional[List[FHIRReference]] = Field(None, description="Intérpretes dos resultados")
    specimen: Optional[List[FHIRReference]] = Field(None, description="Espécimes")
    result: Optional[List[FHIRReference]] = Field(None, description="Resultados")
    imagingStudy: Optional[List[FHIRReference]] = Field(None, description="Estudos de imagem")
    media: Optional[List[Dict[str, Any]]] = Field(None, description="Mídia")
    conclusion: Optional[str] = Field(None, description="Conclusão")
    conclusionCode: Optional[List[FHIRCodeableConcept]] = Field(None, description="Códigos de conclusão")
    presentedForm: Optional[List[Dict[str, Any]]] = Field(None, description="Formulário apresentado")


class FHIRPatient(BaseModel):
    """Paciente FHIR R4."""
    resourceType: str = Field(default="Patient", description="Tipo do recurso")
    id: Optional[str] = Field(None, description="ID lógico do recurso")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadados")
    implicitRules: Optional[str] = Field(None, description="Regras implícitas")
    language: Optional[str] = Field(None, description="Idioma")
    text: Optional[Dict[str, Any]] = Field(None, description="Narrativa textual")
    contained: Optional[List[Dict[str, Any]]] = Field(None, description="Recursos contidos")
    extension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões")
    modifierExtension: Optional[List[Dict[str, Any]]] = Field(None, description="Extensões modificadoras")
    
    # Elementos específicos do Paciente
    identifier: Optional[List[Dict[str, Any]]] = Field(None, description="Identificadores")
    active: Optional[bool] = Field(None, description="Ativo")
    name: Optional[List[Dict[str, Any]]] = Field(None, description="Nomes")
    telecom: Optional[List[Dict[str, Any]]] = Field(None, description="Telecomunicações")
    gender: Optional[str] = Field(None, description="Gênero")
    birthDate: Optional[str] = Field(None, description="Data de nascimento")
    deceasedBoolean: Optional[bool] = Field(None, description="Falecido (booleano)")
    deceasedDateTime: Optional[datetime] = Field(None, description="Falecido (data/hora)")
    address: Optional[List[Dict[str, Any]]] = Field(None, description="Endereços")
    maritalStatus: Optional[FHIRCodeableConcept] = Field(None, description="Estado civil")
    multipleBirthBoolean: Optional[bool] = Field(None, description="Nascimento múltiplo (booleano)")
    multipleBirthInteger: Optional[int] = Field(None, description="Nascimento múltiplo (inteiro)")
    photo: Optional[List[Dict[str, Any]]] = Field(None, description="Fotos")
    contact: Optional[List[Dict[str, Any]]] = Field(None, description="Contatos")
    communication: Optional[List[Dict[str, Any]]] = Field(None, description="Comunicação")
    generalPractitioner: Optional[List[FHIRReference]] = Field(None, description="Médico geral")
    managingOrganization: Optional[FHIRReference] = Field(None, description="Organização responsável")
    link: Optional[List[Dict[str, Any]]] = Field(None, description="Links")


# Utilitários para criação de recursos FHIR
def create_ecg_observation(
    patient_id: str,
    ecg_data: List[float],
    sampling_rate: int,
    analysis_results: Dict[str, Any]
) -> FHIRObservation:
    """Cria observação FHIR para ECG."""
    return FHIRObservation(
        id=f"ecg-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        status=ObservationStatus.FINAL,
        category=[
            FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system="http://terminology.hl7.org/CodeSystem/observation-category",
                        code="procedure",
                        display="Procedure"
                    )
                ]
            )
        ],
        code=FHIRCodeableConcept(
            coding=[
                FHIRCoding(
                    system="http://loinc.org",
                    code="11524-6",
                    display="EKG study"
                )
            ]
        ),
        subject=FHIRReference(
            reference=f"Patient/{patient_id}"
        ),
        effectiveDateTime=datetime.now(),
        valueQuantity=FHIRQuantity(
            value=analysis_results.get("confidence", 0.0),
            unit="confidence_score",
            system="http://unitsofmeasure.org"
        ),
        component=[
            FHIRObservationComponent(
                code=FHIRCodeableConcept(
                    coding=[
                        FHIRCoding(
                            system="http://loinc.org",
                            code="8867-4",
                            display="Heart rate"
                        )
                    ]
                ),
                valueQuantity=FHIRQuantity(
                    value=sampling_rate,
                    unit="Hz",
                    system="http://unitsofmeasure.org"
                )
            )
        ]
    )


def create_ecg_diagnostic_report(
    patient_id: str,
    observations: List[str],
    conclusion: str
) -> FHIRDiagnosticReport:
    """Cria relatório diagnóstico FHIR para ECG."""
    return FHIRDiagnosticReport(
        id=f"ecg-report-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        status=DiagnosticReportStatus.FINAL,
        category=[
            FHIRCodeableConcept(
                coding=[
                    FHIRCoding(
                        system="http://terminology.hl7.org/CodeSystem/v2-0074",
                        code="CG",
                        display="Cardiology"
                    )
                ]
            )
        ],
        code=FHIRCodeableConcept(
            coding=[
                FHIRCoding(
                    system="http://loinc.org",
                    code="11524-6",
                    display="EKG study"
                )
            ]
        ),
        subject=FHIRReference(
            reference=f"Patient/{patient_id}"
        ),
        effectiveDateTime=datetime.now(),
        issued=datetime.now(),
        result=[
            FHIRReference(reference=f"Observation/{obs_id}")
            for obs_id in observations
        ],
        conclusion=conclusion
    )

