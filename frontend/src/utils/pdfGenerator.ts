// PDF Report Generator for CardioAI Pro
// Gerador automático de relatórios médicos em PDF

import jsPDF from 'jspdf'
import 'jspdf-autotable'
import { format } from 'date-fns'
import { ptBR } from 'date-fns/locale'

// Types
interface ReportData {
  patient: {
    name: string
    mrn: string
    dateOfBirth: string
    gender: string
    age: number
  }
  physician: {
    name: string
    license: string
    specialty: string
  }
  facility: {
    name: string
    address: string
    phone: string
  }
  ecg: {
    id: string
    timestamp: string
    duration: number
    heartRate: number
    rhythm: string
    interpretation: string
    abnormalities: string[]
    aiConfidence: number
    technician?: string
  }
  analysis: {
    findings: string[]
    recommendations: string[]
    riskLevel: 'low' | 'medium' | 'high' | 'critical'
    followUp?: string
  }
  images?: {
    ecgWaveform?: string // base64 image
    rhythmStrip?: string // base64 image
  }
}

interface ReportOptions {
  includeImages: boolean
  includeAIAnalysis: boolean
  includeRecommendations: boolean
  watermark?: string
  language: 'pt' | 'en'
}

export class PDFReportGenerator {
  private doc: jsPDF
  private pageWidth: number
  private pageHeight: number
  private margin: number
  private currentY: number

  constructor() {
    this.doc = new jsPDF('p', 'mm', 'a4')
    this.pageWidth = this.doc.internal.pageSize.getWidth()
    this.pageHeight = this.doc.internal.pageSize.getHeight()
    this.margin = 20
    this.currentY = this.margin
  }

  async generateECGReport(data: ReportData, options: ReportOptions = {
    includeImages: true,
    includeAIAnalysis: true,
    includeRecommendations: true,
    language: 'pt'
  }): Promise<Blob> {
    // Reset document
    this.doc = new jsPDF('p', 'mm', 'a4')
    this.currentY = this.margin

    // Add header
    this.addHeader(data.facility)

    // Add patient information
    this.addPatientInfo(data.patient, options.language)

    // Add ECG data
    this.addECGData(data.ecg, options.language)

    // Add ECG images if available
    if (options.includeImages && data.images) {
      this.addECGImages(data.images)
    }

    // Add AI analysis
    if (options.includeAIAnalysis) {
      this.addAIAnalysis(data.ecg, options.language)
    }

    // Add clinical findings
    this.addClinicalFindings(data.analysis, options.language)

    // Add recommendations
    if (options.includeRecommendations) {
      this.addRecommendations(data.analysis, options.language)
    }

    // Add physician signature
    this.addPhysicianSignature(data.physician, options.language)

    // Add footer
    this.addFooter(options.language)

    // Add watermark if specified
    if (options.watermark) {
      this.addWatermark(options.watermark)
    }

    return this.doc.output('blob')
  }

  private addHeader(facility: any): void {
    // Logo placeholder
    this.doc.setFillColor(59, 130, 246) // Blue
    this.doc.rect(this.margin, this.currentY, 30, 15, 'F')
    
    this.doc.setTextColor(255, 255, 255)
    this.doc.setFontSize(12)
    this.doc.setFont('helvetica', 'bold')
    this.doc.text('CardioAI', this.margin + 15, this.currentY + 9, { align: 'center' })

    // Facility info
    this.doc.setTextColor(0, 0, 0)
    this.doc.setFontSize(16)
    this.doc.setFont('helvetica', 'bold')
    this.doc.text(facility.name, this.margin + 35, this.currentY + 6)
    
    this.doc.setFontSize(10)
    this.doc.setFont('helvetica', 'normal')
    this.doc.text(facility.address, this.margin + 35, this.currentY + 11)
    this.doc.text(facility.phone, this.margin + 35, this.currentY + 15)

    // Report title
    this.doc.setFontSize(18)
    this.doc.setFont('helvetica', 'bold')
    this.doc.setTextColor(59, 130, 246)
    this.doc.text('RELATÓRIO DE ELETROCARDIOGRAMA', this.pageWidth / 2, this.currentY + 25, { align: 'center' })

    this.currentY += 40
  }

  private addPatientInfo(patient: any, language: string): void {
    const labels = language === 'pt' ? {
      patientInfo: 'INFORMAÇÕES DO PACIENTE',
      name: 'Nome',
      mrn: 'Registro',
      dob: 'Data de Nascimento',
      gender: 'Sexo',
      age: 'Idade'
    } : {
      patientInfo: 'PATIENT INFORMATION',
      name: 'Name',
      mrn: 'MRN',
      dob: 'Date of Birth',
      gender: 'Gender',
      age: 'Age'
    }

    this.addSectionTitle(labels.patientInfo)

    const patientData = [
      [labels.name, patient.name],
      [labels.mrn, patient.mrn],
      [labels.dob, format(new Date(patient.dateOfBirth), 'dd/MM/yyyy', { locale: ptBR })],
      [labels.gender, patient.gender === 'M' ? 'Masculino' : 'Feminino'],
      [labels.age, `${patient.age} anos`]
    ]

    ;(this.doc as any).autoTable({
      startY: this.currentY,
      head: [],
      body: patientData,
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: 2
      },
      columnStyles: {
        0: { fontStyle: 'bold', cellWidth: 40 },
        1: { cellWidth: 80 }
      }
    })

    this.currentY = (this.doc as any).lastAutoTable.finalY + 10
  }

  private addECGData(ecg: any, language: string): void {
    const labels = language === 'pt' ? {
      ecgData: 'DADOS DO ECG',
      examId: 'ID do Exame',
      datetime: 'Data/Hora',
      duration: 'Duração',
      heartRate: 'Frequência Cardíaca',
      rhythm: 'Ritmo',
      technician: 'Técnico'
    } : {
      ecgData: 'ECG DATA',
      examId: 'Exam ID',
      datetime: 'Date/Time',
      duration: 'Duration',
      heartRate: 'Heart Rate',
      rhythm: 'Rhythm',
      technician: 'Technician'
    }

    this.addSectionTitle(labels.ecgData)

    const ecgData = [
      [labels.examId, ecg.id],
      [labels.datetime, format(new Date(ecg.timestamp), 'dd/MM/yyyy HH:mm:ss', { locale: ptBR })],
      [labels.duration, `${ecg.duration} segundos`],
      [labels.heartRate, `${ecg.heartRate} bpm`],
      [labels.rhythm, ecg.rhythm],
      ...(ecg.technician ? [[labels.technician, ecg.technician]] : [])
    ]

    ;(this.doc as any).autoTable({
      startY: this.currentY,
      head: [],
      body: ecgData,
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: 2
      },
      columnStyles: {
        0: { fontStyle: 'bold', cellWidth: 40 },
        1: { cellWidth: 80 }
      }
    })

    this.currentY = (this.doc as any).lastAutoTable.finalY + 10
  }

  private addECGImages(images: any): void {
    if (images.ecgWaveform) {
      this.addSectionTitle('TRAÇADO ECG')
      
      try {
        this.doc.addImage(
          images.ecgWaveform,
          'PNG',
          this.margin,
          this.currentY,
          this.pageWidth - 2 * this.margin,
          60
        )
        this.currentY += 70
      } catch (error) {
        console.error('Error adding ECG waveform image:', error)
      }
    }

    if (images.rhythmStrip) {
      this.addSectionTitle('FAIXA DE RITMO')
      
      try {
        this.doc.addImage(
          images.rhythmStrip,
          'PNG',
          this.margin,
          this.currentY,
          this.pageWidth - 2 * this.margin,
          30
        )
        this.currentY += 40
      } catch (error) {
        console.error('Error adding rhythm strip image:', error)
      }
    }
  }

  private addAIAnalysis(ecg: any, language: string): void {
    const labels = language === 'pt' ? {
      aiAnalysis: 'ANÁLISE DE INTELIGÊNCIA ARTIFICIAL',
      interpretation: 'Interpretação',
      confidence: 'Confiança da IA',
      abnormalities: 'Anormalidades Detectadas'
    } : {
      aiAnalysis: 'ARTIFICIAL INTELLIGENCE ANALYSIS',
      interpretation: 'Interpretation',
      confidence: 'AI Confidence',
      abnormalities: 'Detected Abnormalities'
    }

    this.addSectionTitle(labels.aiAnalysis)

    // AI interpretation
    this.doc.setFontSize(10)
    this.doc.setFont('helvetica', 'bold')
    this.doc.text(`${labels.interpretation}:`, this.margin, this.currentY)
    
    this.doc.setFont('helvetica', 'normal')
    const interpretationLines = this.doc.splitTextToSize(ecg.interpretation, this.pageWidth - 2 * this.margin)
    this.doc.text(interpretationLines, this.margin, this.currentY + 5)
    this.currentY += interpretationLines.length * 4 + 10

    // AI confidence
    this.doc.setFont('helvetica', 'bold')
    this.doc.text(`${labels.confidence}:`, this.margin, this.currentY)
    
    this.doc.setFont('helvetica', 'normal')
    const confidenceColor = ecg.aiConfidence >= 0.9 ? [34, 197, 94] : 
                           ecg.aiConfidence >= 0.7 ? [251, 191, 36] : [239, 68, 68]
    this.doc.setTextColor(confidenceColor[0], confidenceColor[1], confidenceColor[2])
    this.doc.text(`${(ecg.aiConfidence * 100).toFixed(1)}%`, this.margin + 30, this.currentY)
    this.doc.setTextColor(0, 0, 0)
    this.currentY += 10

    // Abnormalities
    if (ecg.abnormalities && ecg.abnormalities.length > 0) {
      this.doc.setFont('helvetica', 'bold')
      this.doc.text(`${labels.abnormalities}:`, this.margin, this.currentY)
      this.currentY += 5

      ecg.abnormalities.forEach((abnormality: string, _index: number) => {
        this.doc.setFont('helvetica', 'normal')
        this.doc.text(`• ${abnormality}`, this.margin + 5, this.currentY)
        this.currentY += 5
      })
    }

    this.currentY += 10
  }

  private addClinicalFindings(analysis: any, language: string): void {
    const labels = language === 'pt' ? {
      findings: 'ACHADOS CLÍNICOS',
      riskLevel: 'Nível de Risco'
    } : {
      findings: 'CLINICAL FINDINGS',
      riskLevel: 'Risk Level'
    }

    this.addSectionTitle(labels.findings)

    // Risk level
    this.doc.setFontSize(10)
    this.doc.setFont('helvetica', 'bold')
    this.doc.text(`${labels.riskLevel}:`, this.margin, this.currentY)
    
    const riskColors: Record<string, [number, number, number]> = {
      low: [34, 197, 94],
      medium: [251, 191, 36],
      high: [249, 115, 22],
      critical: [239, 68, 68]
    }
    
    const riskLabels: Record<string, string> = {
      low: 'Baixo',
      medium: 'Médio',
      high: 'Alto',
      critical: 'Crítico'
    }

    this.doc.setTextColor(...riskColors[analysis.riskLevel])
    this.doc.text(riskLabels[analysis.riskLevel], this.margin + 30, this.currentY)
    this.doc.setTextColor(0, 0, 0)
    this.currentY += 10

    // Findings
    if (analysis.findings && analysis.findings.length > 0) {
      analysis.findings.forEach((finding: string) => {
        this.doc.setFont('helvetica', 'normal')
        const findingLines = this.doc.splitTextToSize(`• ${finding}`, this.pageWidth - 2 * this.margin - 5)
        this.doc.text(findingLines, this.margin + 5, this.currentY)
        this.currentY += findingLines.length * 4 + 2
      })
    }

    this.currentY += 10
  }

  private addRecommendations(analysis: any, language: string): void {
    const labels = language === 'pt' ? {
      recommendations: 'RECOMENDAÇÕES',
      followUp: 'Acompanhamento'
    } : {
      recommendations: 'RECOMMENDATIONS',
      followUp: 'Follow-up'
    }

    this.addSectionTitle(labels.recommendations)

    if (analysis.recommendations && analysis.recommendations.length > 0) {
      analysis.recommendations.forEach((recommendation: string) => {
        this.doc.setFont('helvetica', 'normal')
        this.doc.setFontSize(10)
        const recLines = this.doc.splitTextToSize(`• ${recommendation}`, this.pageWidth - 2 * this.margin - 5)
        this.doc.text(recLines, this.margin + 5, this.currentY)
        this.currentY += recLines.length * 4 + 2
      })
    }

    if (analysis.followUp) {
      this.currentY += 5
      this.doc.setFont('helvetica', 'bold')
      this.doc.text(`${labels.followUp}:`, this.margin, this.currentY)
      
      this.doc.setFont('helvetica', 'normal')
      const followUpLines = this.doc.splitTextToSize(analysis.followUp, this.pageWidth - 2 * this.margin)
      this.doc.text(followUpLines, this.margin, this.currentY + 5)
      this.currentY += followUpLines.length * 4 + 10
    }
  }

  private addPhysicianSignature(physician: any, language: string): void {
    const labels = language === 'pt' ? {
      physician: 'MÉDICO RESPONSÁVEL',
      name: 'Nome',
      license: 'CRM',
      specialty: 'Especialidade',
      signature: 'Assinatura Digital',
      date: 'Data do Relatório'
    } : {
      physician: 'ATTENDING PHYSICIAN',
      name: 'Name',
      license: 'License',
      specialty: 'Specialty',
      signature: 'Digital Signature',
      date: 'Report Date'
    }

    // Check if we need a new page
    if (this.currentY > this.pageHeight - 80) {
      this.doc.addPage()
      this.currentY = this.margin
    }

    this.addSectionTitle(labels.physician)

    const physicianData = [
      [labels.name, physician.name],
      [labels.license, physician.license],
      [labels.specialty, physician.specialty]
    ]

    ;(this.doc as any).autoTable({
      startY: this.currentY,
      head: [],
      body: physicianData,
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: 2
      },
      columnStyles: {
        0: { fontStyle: 'bold', cellWidth: 40 },
        1: { cellWidth: 80 }
      }
    })

    this.currentY = (this.doc as any).lastAutoTable.finalY + 15

    // Signature line
    this.doc.setDrawColor(0, 0, 0)
    this.doc.line(this.margin, this.currentY, this.margin + 80, this.currentY)
    
    this.doc.setFontSize(8)
    this.doc.setFont('helvetica', 'normal')
    this.doc.text(labels.signature, this.margin, this.currentY + 5)

    // Date
    this.doc.text(`${labels.date}: ${format(new Date(), 'dd/MM/yyyy HH:mm', { locale: ptBR })}`, 
                  this.pageWidth - this.margin - 50, this.currentY + 5)
  }

  private addFooter(language: string): void {
    const footerText = language === 'pt' ? 
      'Este relatório foi gerado automaticamente pelo sistema CardioAI Pro' :
      'This report was automatically generated by CardioAI Pro system'

    this.doc.setFontSize(8)
    this.doc.setFont('helvetica', 'italic')
    this.doc.setTextColor(128, 128, 128)
    this.doc.text(footerText, this.pageWidth / 2, this.pageHeight - 10, { align: 'center' })
  }

  private addWatermark(text: string): void {
    const pageCount = this.doc.getNumberOfPages()
    
    for (let i = 1; i <= pageCount; i++) {
      this.doc.setPage(i)
      this.doc.setFontSize(50)
      this.doc.setFont('helvetica', 'bold')
      this.doc.setTextColor(200, 200, 200)
      
      // Rotate and center the watermark
      this.doc.text(text, this.pageWidth / 2, this.pageHeight / 2, {
        align: 'center',
        angle: 45
      })
    }
  }

  private addSectionTitle(title: string): void {
    // Check if we need a new page
    if (this.currentY > this.pageHeight - 40) {
      this.doc.addPage()
      this.currentY = this.margin
    }

    this.doc.setFontSize(12)
    this.doc.setFont('helvetica', 'bold')
    this.doc.setTextColor(59, 130, 246)
    this.doc.text(title, this.margin, this.currentY)
    
    // Add underline
    this.doc.setDrawColor(59, 130, 246)
    this.doc.line(this.margin, this.currentY + 2, this.margin + this.doc.getTextWidth(title), this.currentY + 2)
    
    this.doc.setTextColor(0, 0, 0)
    this.currentY += 10
  }
}

// Export functions
export const generateECGReport = async (data: ReportData, options?: ReportOptions): Promise<Blob> => {
  const generator = new PDFReportGenerator()
  return generator.generateECGReport(data, options)
}

export const downloadReport = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export default PDFReportGenerator

