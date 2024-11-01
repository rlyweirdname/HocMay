'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Wand2, 
  Eye, 
  Sparkles, 
  Copy, 
  CheckCircle,
  X 
} from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface Suggestion {
  text: string
  probability: number
}

interface Toast {
  id: number
  title: string
  description: string
  type: 'success' | 'error'
}

export default function BERTTextGenerator() {
  const [template, setTemplate] = useState<string>('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [attentionHeatmap, setAttentionHeatmap] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])

  const showToast = (title: string, description: string, type: 'success' | 'error') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, title, description, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(toast => toast.id !== id))
    }, 3000)
  }

  const generateText = async () => {
    if (!template.trim()) {
      showToast("Error", "Please enter a template with [MASK]", "error")
      return
    }

    setLoading(true)
    setSuggestions([])
    setAttentionHeatmap('')

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate-text/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          template: template,
          num_suggestions: 3
        })
      })
      
      if (!response.ok) {
        throw new Error('Failed to generate text')
      }
      
      const data = await response.json()
      setSuggestions(data.suggestions)
    } catch (error) {
      showToast("Error", "Failed to generate text suggestions", "error")
      console.error('Error generating text:', error)
    } finally {
      setLoading(false)
    }
  }

  const visualizeAttention = async () => {
    if (!template.trim()) {
      showToast("Error", "Please enter a sentence to analyze", "error")
      return
    }

    setLoading(true)
    setSuggestions([])
    setAttentionHeatmap('')

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/visualize-attention/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sentence: template })
      })
      
      if (!response.ok) {
        throw new Error('Failed to generate heatmap')
      }
      
      const data = await response.json()
      setAttentionHeatmap(`data:image/png;base64,${data.attention_heatmap}`)
    } catch (error) {
      showToast("Error", "Failed to generate attention heatmap", "error")
      console.error('Error visualizing attention:', error)
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(index)
    showToast("Copied", "Text suggestion copied to clipboard", "success")
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-8">
      {/* Toast notifications */}
      <div className="fixed top-4 right-4 z-50">
        <AnimatePresence>
          {toasts.map(toast => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`mb-2 p-4 rounded-lg shadow-lg flex items-center justify-between ${
                toast.type === 'error' ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
              }`}
            >
              <div>
                <h4 className="font-semibold">{toast.title}</h4>
                <p className="text-sm">{toast.description}</p>
              </div>
              <button
                onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))}
                className="ml-4 p-1 hover:bg-white/20 rounded"
              >
                <X size={16} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <motion.div 
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-2xl mx-auto bg-white dark:bg-gray-800 shadow-xl rounded-lg p-6"
      >
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-900 dark:text-white">
          BERT Text Insights
        </h1>

        <Textarea 
          value={template}
          onChange={(e) => setTemplate(e.target.value)}
          placeholder="Enter a sentence with [MASK] or a sentence to analyze"
          className="mb-4"
        />

        <div className="flex justify-center gap-4 mb-6">
          <Button 
            onClick={generateText}
            disabled={loading}
            className="inline-flex items-center"
            variant="outline"
          >
            <Wand2 className="mr-2 h-4 w-4" /> Generate Text
          </Button>

          <Button 
            onClick={visualizeAttention}
            disabled={loading}
            className="inline-flex items-center"
            variant="secondary"
          >
            <Eye className="mr-2 h-4 w-4" /> Visualize Attention
          </Button>
        </div>

        {loading && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-center items-center"
          >
            <Sparkles className="animate-spin text-purple-500" size={32} />
            <span className="ml-2 text-gray-600 dark:text-gray-300">Processing...</span>
          </motion.div>
        )}

        <AnimatePresence>
          {suggestions.length > 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6"
            >
              <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200">
                Text Suggestions
              </h2>
              {suggestions.map((suggestion, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg mb-3 flex justify-between items-center"
                >
                  <div>
                    <p className="font-medium dark:text-white">{suggestion.text}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Probability: {(suggestion.probability * 100).toFixed(2)}%
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => copyToClipboard(suggestion.text, index)}
                  >
                    {copiedIndex === index ? (
                      <CheckCircle className="text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </motion.div>
              ))}
            </motion.div>
          )}

          {attentionHeatmap && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6"
            >
              <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-200">
                Attention Heatmap
              </h2>
              <motion.img 
                src={attentionHeatmap} 
                alt="BERT Attention Heatmap"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="w-full rounded-lg shadow-md"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}