// frontend/src/pages/ECGImageAnalysisPage.tsx

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { setECGData, setLoading, setError } from '../store/slices/ecgSlice';
import { digitizeECGImage, DigitizedECGData } from '../services/medicalAPI';
import { ModernECGVisualization } from '../components/medical/ModernECGVisualization';
import { AIInsightPanel } from '../components/ui/AIInsightPanel';
import { PhotoIcon, ArrowUpTrayIcon, XCircleIcon } from '@heroicons/react/24/solid';

const ECGImageAnalysisPage: React.FC = () => {
  const dispatch = useDispatch();
  const { currentECG, loading, error: analysisError } = useSelector((state: RootState) => state.ecg);
  
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: any[]) => {
    // Limpa estados anteriores
    setUploadError(null);
    dispatch(setError(null));
    dispatch(setECGData(null));
    
    if (fileRejections.length > 0) {
        setUploadError("Arquivo inválido. Por favor, envie apenas imagens PNG ou JPG.");
        setUploadedFile(null);
        setPreview(null);
        return;
    }

    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  }, [dispatch]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    multiple: false,
  });

  const handleDigitize = async () => {
    if (!uploadedFile) {
      setUploadError('Por favor, selecione uma imagem primeiro.');
      return;
    }

    setIsProcessing(true);
    dispatch(setLoading(true));
    dispatch(setError(null));
    setUploadError(null);

    try {
      const result: DigitizedECGData = await digitizeECGImage(uploadedFile);
      dispatch(setECGData({
        id: `image-${new Date().getTime()}`,
        data: result.signal_data,
        sampling_rate: result.sampling_rate,
        leads: result.lead_names,
      }));
    } catch (error: any) {
      const message = error.message || 'Ocorreu um erro desconhecido.';
      dispatch(setError(message));
    } finally {
      setIsProcessing(false);
      dispatch(setLoading(false));
    }
  };
  
  const clearSelection = () => {
      setUploadedFile(null);
      setPreview(null);
      dispatch(setECGData(null));
      dispatch(setError(null));
      setUploadError(null);
  }

  return (
    <div className="container mx-auto p-4 md:p-6 space-y-6">
      <div className="bg-white p-6 rounded-xl shadow-lg">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Análise de Imagem de ECG</h2>
        <div
          {...getRootProps()}
          className={`relative p-8 border-2 border-dashed rounded-lg text-center cursor-pointer transition-all duration-300
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center justify-center space-y-4 text-gray-500">
            <ArrowUpTrayIcon className="h-12 w-12" />
            {preview ? (
              <div className="relative">
                <img src={preview} alt="Pré-visualização do ECG" className="max-h-60 mx-auto rounded-md shadow-md" />
              </div>
            ) : (
              <div>
                <p className="font-semibold">Arraste e solte uma imagem aqui</p>
                <p className="text-sm">ou clique para selecionar (PNG, JPG)</p>
              </div>
            )}
          </div>
        </div>

        {uploadError && (
          <div className="mt-4 flex items-center justify-center text-red-600">
            <XCircleIcon className="h-5 w-5 mr-2" />
            <span>{uploadError}</span>
          </div>
        )}

        {uploadedFile && (
          <div className="mt-4 flex flex-col items-center space-y-3">
            <p className="text-sm text-gray-600">Arquivo: <span className="font-medium">{uploadedFile.name}</span></p>
            <div className="flex space-x-4">
                 <button
                    onClick={clearSelection}
                    className="px-5 py-2 bg-gray-200 text-gray-700 font-semibold rounded-lg shadow-sm hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400"
                  >
                    Remover
                  </button>
                <button
                    onClick={handleDigitize}
                    disabled={isProcessing}
                    className="px-5 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {isProcessing ? 'Processando...' : 'Digitalizar e Analisar'}
                </button>
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div className="text-center p-4 bg-white rounded-xl shadow-lg">
          <p className="text-lg text-blue-600 animate-pulse">Analisando sinal... Por favor, aguarde.</p>
        </div>
      )}

      {analysisError && !loading && (
         <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md shadow-lg" role="alert">
           <p className="font-bold">Erro na Análise</p>
           <p>{analysisError}</p>
         </div>
      )}

      {currentECG && !loading && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-white p-4 rounded-xl shadow-lg">
             <h3 className="text-xl font-bold mb-4 text-gray-800">Sinal Digitalizado</h3>
            <ModernECGVisualization
              ecgData={currentECG.data}
              samplingRate={currentECG.sampling_rate}
              leadNames={currentECG.leads}
            />
          </div>
          <div className="lg:col-span-1">
            <AIInsightPanel ecgId={currentECG.id} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ECGImageAnalysisPage;

