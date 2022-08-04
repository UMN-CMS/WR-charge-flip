
// -*- C++ -*-
//
// Package:    ExoAnalysis/NNstudies
// Class:      NNstudies
//
/**\class NNstudies NNstudies.cc ExoAnalysis/NNstudies/plugins/NNstudies.cc
 Description: cut flow, event selection, and extraction of variables for NN
 Implementation:
     [Notes on implementation]
*/
//
//  Author:  Andrew Evans, adapted by Martin Meier
//         Created:  Aug 2021
//
//

// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "Math/GenVector/Boost.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"

#include "HEEP/VID/interface/CutNrs.h"
#include "HEEP/VID/interface/VIDCutCodes.h"

#include "TLorentzVector.h"
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TVectorD.h>
#include <cmath>

#include "TH1.h"
#include "TMath.h"
#include "TDirectory.h"


#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "ExoAnalysis/WR_lite/interface/eventInfo.h"
#include "ExoAnalysis/WR_lite/interface/cutFlowHistos.h"


//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.


using reco::TrackCollection;

class NNstudies : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
	public:
		explicit NNstudies(const edm::ParameterSet&);
		~NNstudies();

		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


	private:
		virtual void beginJob() override;
		virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
		virtual void endJob() override;
		double dR2(double eta1, double eta2, double phi1, double phi2);
		double dPhi(double phi1, double phi2);
		bool tWfinder(const edm::Event&, const reco::GenParticle* );
		bool tfinder(const edm::Event&, const reco::GenParticle* );
		bool passElectronTrig(const edm::Event&);
		bool electronMVAcut(float pt, float eta, float bdt);
		void csvTable(const reco::GenParticle*, const reco::GenParticle*, const reco::GenParticle*, const reco::GenParticle*, const reco::GenParticle*, const reco::GenParticle*, int binNumber, const pat::Muon*, const reco::GsfElectron*, const reco::GenJet*, const reco::GenJet*, math::XYZTLorentzVector combinedGenJets, const pat::Jet*, const pat::Jet*, math::XYZTLorentzVector combinedJets, const pat::MET, double weight, int ematches);
		void countTable(int count);
		int binNumber(const reco::GenParticle*);
		
		
		cutFlowHistos m_histoMaker;

		TH1D* m_eventsWeight;

		// ----------member data ---------------------------

		edm::EDGetTokenT<TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
		edm::EDGetToken m_genParticleToken;
		edm::EDGetToken m_recoMETToken;
		edm::EDGetToken m_highMuonToken;
		edm::EDGetToken m_highElectronToken;
		edm::EDGetToken m_AK4genCHSJetsToken;
		edm::EDGetToken m_AK4CHSJetsToken;
		edm::EDGetToken m_packedGenParticlesToken;
		edm::EDGetToken m_packedPFCandidatesToken;
		edm::EDGetToken m_genEventInfoToken;
		edm::EDGetToken m_offlineVerticesToken;
		std::vector<std::string>  m_electronPathsToPass;
		edm::EDGetToken m_trigResultsToken;
		
		std::string m_dataSaveFile;
		edm::EDGetToken rhoToken;

		bool m_genTrainData;

		edm::EDGetTokenT<edm::ValueMap<float> > mvaValuesMapToken_;

		std::string  cSV_bTag1      = "pfDeepCSVJetTags:probb";
		std::string  cSV_bTag2      = "pfDeepCSVJetTags:probbb";

		double binEdges[17] = {0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 300.0, 350.0, 400.0, 1000.0};
		
		edm::Service<TFileService> fs; 
		
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
NNstudies::NNstudies(const edm::ParameterSet& iConfig)
	:
	tracksToken_(consumes<TrackCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
	m_genParticleToken(consumes<std::vector<reco::GenParticle>> (iConfig.getParameter<edm::InputTag>("genParticles"))),
	m_recoMETToken(consumes<std::vector<pat::MET>> (iConfig.getParameter<edm::InputTag>("recoMET"))),
	m_highMuonToken (consumes<std::vector<pat::Muon>> (iConfig.getParameter<edm::InputTag>("highMuons"))),
	m_highElectronToken (consumes<edm::View<reco::GsfElectron>> (iConfig.getParameter<edm::InputTag>("highElectrons"))),
	m_AK4genCHSJetsToken (consumes<std::vector<reco::GenJet>> (iConfig.getParameter<edm::InputTag>("AK4genCHSJets"))),
	m_AK4CHSJetsToken (consumes<std::vector<pat::Jet>> (iConfig.getParameter<edm::InputTag>("AK4CHSJets"))),
	m_packedGenParticlesToken (consumes<std::vector<pat::PackedGenParticle>> (iConfig.getParameter<edm::InputTag>("packedGenParticles"))),
	m_packedPFCandidatesToken (consumes<std::vector<pat::PackedCandidate>> (iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
	m_genEventInfoToken (consumes<GenEventInfoProduct> (iConfig.getParameter<edm::InputTag>("genInfo"))),
	m_offlineVerticesToken (consumes<std::vector<reco::Vertex>> (iConfig.getParameter<edm::InputTag>("vertices"))),
	m_dataSaveFile (iConfig.getUntrackedParameter<std::string>("trainFile")),
	rhoToken (consumes<double> (iConfig.getParameter<edm::InputTag>("rho"))),
	mvaValuesMapToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("mvaValuesMap")))
{
   //now do what ever initialization is needed

   m_electronPathsToPass  = iConfig.getParameter<std::vector<std::string> >("electronPathsToPass");
   m_trigResultsToken = consumes<edm::TriggerResults> (iConfig.getParameter<edm::InputTag>("trigResults"));
}


NNstudies::~NNstudies()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void
NNstudies::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	eventBits2 iBit; 
    eventInfo myEvent; 
	
	edm::Handle<GenEventInfoProduct> eventInfo;
	iEvent.getByToken(m_genEventInfoToken, eventInfo);

	edm::Handle<std::vector<pat::Muon>> highMuons;
	iEvent.getByToken(m_highMuonToken, highMuons);

	edm::Handle<edm::View<reco::GsfElectron> > highElectrons;
	iEvent.getByToken(m_highElectronToken, highElectrons);

	edm::Handle<std::vector<reco::GenParticle>> genParticles;
	iEvent.getByToken(m_genParticleToken, genParticles);

	edm::Handle<std::vector<pat::PackedGenParticle>> packedGenParticles;
	iEvent.getByToken(m_packedGenParticlesToken, packedGenParticles);

	edm::Handle<std::vector<pat::PackedCandidate>> packedPFCandidates;
	iEvent.getByToken(m_packedPFCandidatesToken, packedPFCandidates);

	edm::Handle<std::vector<pat::MET>> recoMET;
	iEvent.getByToken(m_recoMETToken, recoMET);

	edm::Handle<edm::ValueMap<float> > mvaValues;
    iEvent.getByToken(mvaValuesMapToken_,mvaValues);

    edm::Handle<double> rhotoken;
    iEvent.getByLabel("fixedGridRhoFastjetAll", rhotoken);
    double rho = *(rhotoken.product());
	
	float eventCount = eventInfo->weight()/fabs(eventInfo->weight());
	
	edm::Handle<std::vector<reco::Vertex>> vertices;
	iEvent.getByToken(m_offlineVerticesToken, vertices);
	if(!vertices.isValid()) {
		throw cms::Exception("Vertex collection not valid!");
	}

	iBit.hasPVertex = myEvent.PVselection(vertices);


	// gen lepton info

	const reco::GenParticle* genMuon = 0;
	const reco::GenParticle* genElectron = 0;
	const reco::GenParticle* muNu = 0;
	const reco::GenParticle* eNu = 0;

	const reco::GenParticle* bquark = 0;
	const reco::GenParticle* antibquark = 0;
	const reco::GenParticle* tquark = 0;
	const reco::GenParticle* antitquark = 0;

	for (std::vector<reco::GenParticle>::const_iterator iParticle = genParticles->begin(); iParticle != genParticles->end(); iParticle++) {
		if( ! iParticle->isHardProcess() ){ continue; }
		if(iParticle->pdgId()==6 && tquark==0){ tquark =  &(*(iParticle)); }
		if(iParticle->pdgId()==-6 && antitquark==0){ antitquark =  &(*(iParticle)); }
		if(tfinder(iEvent, &(*iParticle))){ // check if the gen particle comes from a top
			if(iParticle->pdgId()==5){ bquark = &(*(iParticle)); }
			if(iParticle->pdgId()==-5){ antibquark = &(*(iParticle)); }

			if(tWfinder(iEvent, &(*iParticle))){  // check if the gen particle comes from a top->W->lepton
				if(abs(iParticle->pdgId())==13){
					if(genMuon==0){genMuon = &(*(iParticle));}
				}	
				else if(abs(iParticle->pdgId())==11){
					// leptonCount += 1;
					if(genElectron==0){genElectron = &(*(iParticle));}
				}
				else if(abs(iParticle->pdgId())==12){
					if(eNu==0){eNu = &(*(iParticle));}
				}
				else if(abs(iParticle->pdgId())==14){
					if(muNu==0){muNu = &(*(iParticle));}
				}

			}
		}
	}


	//start reco

	const pat::MET Met = recoMET->front();

	math::XYZTLorentzVector combinedGenJetsP4 = {0., 0., 0., 0.};
	math::XYZTLorentzVector combinedJetsP4 = {0., 0., 0., 0.};

	edm::Handle<std::vector<reco::GenJet>> genJets;  
	iEvent.getByToken(m_AK4genCHSJetsToken, genJets);  

	edm::Handle<std::vector<pat::Jet>> JetsAK4;  
	iEvent.getByToken(m_AK4CHSJetsToken, JetsAK4);  

	const reco::GenJet* bGenJet=0;
	const reco::GenJet* antibGenJet=0;

	const reco::GenJet* muJet = 0;
	const reco::GenJet* eJet = 0;

	const pat::Jet* bJet=0;
	const pat::Jet* Jet=0;


    for( std::vector<reco::GenJet>::const_iterator iJet = genJets->begin(); iJet!= genJets->end(); iJet++) {

      if(genMuon!=0 && genElectron!=0 && bquark !=0 && antibquark != 0){
      if(sqrt(dR2(iJet->eta(), genMuon->eta(), iJet->phi(), genMuon->phi())) > 0.35 && sqrt(dR2(iJet->eta(), genElectron->eta(), iJet->phi(), genElectron->phi())) > 0.35 ){
        if (bGenJet == 0 && sqrt(dR2(iJet->eta(), bquark->eta(), iJet->phi(), bquark->phi())) < 0.3  ){
        	bGenJet = &(*(iJet));
        }
        else if(antibGenJet == 0 && sqrt(dR2(iJet->eta(), antibquark->eta(), iJet->phi(), antibquark->phi())) < 0.3 ){
        	antibGenJet = &(*(iJet));
        }
        else{
        	combinedGenJetsP4 = combinedGenJetsP4 + iJet->p4();
        }
      }
  	  }

    }

	// check electron trigger
	//if (passElectronTrig(iEvent)){ electronTrigger=true; }

	//muon/electron reconstruction

		//muon reco

	const pat::Muon* recoMuon=0;

   	for(std::vector<pat::Muon>::const_iterator iMuon = highMuons->begin(); iMuon != highMuons->end(); iMuon++){

   		if(!(iMuon->isHighPtMuon(*myEvent.PVertex))) continue; // || !iMuon->passed(reco::Muon::TkIsoTight)) continue; //preliminary cut

   		if(recoMuon==0 && genMuon!=0){
   			if(sqrt(dR2(iMuon->eta(), genMuon->eta(), iMuon->phi(), genMuon->phi())) < 0.3){
   				recoMuon=&(*(iMuon));
   			}
   		}
   		
	}

		//electron reco

	const reco::GsfElectron* recoElectron=0;
	int ematches = 0;

		for (size_t i = 0; i < highElectrons->size(); ++i){    
				const auto iElectron = highElectrons->ptrAt(i);
				// const bool tightID = iElectron->electronID("cutBasedElectronID-Fall17-94X-V2-tight");
				// if(tightID == false){continue;}

				if(genElectron!=0){
					if(sqrt(dR2(iElectron->eta(), genElectron->eta(), iElectron->phi(), genElectron->phi())) > 0.3) continue;
				}

				//mva cuts
				double bdt = (*mvaValues)[iElectron];
				double pt = iElectron->pt();
				double eta = abs(iElectron->eta());
				if (!(electronMVAcut(pt, eta, bdt))) continue;
				
				//isolation
				double R = 0;
				if (pt > 50){
					if(pt < 200){
						R = 10/pt;
					}
					else{ 
						R = 10/200;
					}
				}
				else if(pt < 50){
					R = 10/50;
				}

				double area = 0;
				if(eta<1.0)        area = 0.1440;
				else if(eta<1.479) area = 0.1562;
				else if(eta<2.0)   area = 0.1032;
				else if(eta<2.2)   area = 0.0859;
				else if(eta<2.3)   area = 0.1116;
				else if(eta<2.4)   area = 0.1321;
				else if(eta<2.5)   area = 0.1654;


				double chargeSum = 0;
				double neutralSum = 0;
				double photonSum = 0;
				for (std::vector<pat::PackedCandidate>::const_iterator iParticle = packedPFCandidates->begin(); iParticle != packedPFCandidates->end(); iParticle++){
					double dr = sqrt(dR2(iElectron->eta(), iParticle->eta(), iElectron->phi(), iParticle->phi()));
					int id = iParticle->pdgId();
					if(dr < R){
						if(id == 22){
							photonSum += iParticle->pt();
						}
						else if(id == 130){
							neutralSum += iParticle->pt();
						}
						else if(id == 211){
							chargeSum += iParticle->pt();
						}
					}

				}
				
				double Imini = 0;
				if(0.0 > (neutralSum + photonSum + rho * area * pow((R/0.3),2.0))){
					Imini = chargeSum / iElectron->pt();
				}
				else{
					Imini = (chargeSum - neutralSum + photonSum + rho * area * pow((R/0.3),2.0)) / iElectron->pt();
				}

				double jetdR = 1000;
				double newjetdR;
				const pat::Jet* electronJet = 0;
				for(std::vector<pat::Jet>::const_iterator iJet = JetsAK4->begin(); iJet != JetsAK4->end(); iJet++) {
					newjetdR = sqrt(dR2(iElectron->eta(), iJet->eta(), iElectron->phi(), iJet->phi()));
					if (newjetdR < jetdR){
						jetdR = newjetdR;
						electronJet = &(*(iJet));
					}
				}

				double p_ratio = iElectron->pt() / electronJet->pt();

				double p_rel = (electronJet->p4() - iElectron->p4()).Dot(iElectron->p4()) / (electronJet->p4() - iElectron->p4()).mag2();

				if (Imini < 0.07 && (p_ratio > 0.78 || p_rel > 8.0)){ //passes isolation
					ematches+=1;
					if(recoElectron==0) {
						recoElectron=&(*(iElectron));
					}
					
				}
				
		}

	//Get reco jets

	for(std::vector<pat::Jet>::const_iterator iJet = JetsAK4->begin(); iJet != JetsAK4->end(); iJet++) {

		double NHF  =           iJet->neutralHadronEnergyFraction(); //cuts 1
		double NEMF =           iJet->neutralEmEnergyFraction(); //cuts 2
		double CHF  =           iJet->chargedHadronEnergyFraction(); //cuts 3
		double CEMF =           iJet->chargedEmEnergyFraction();  //cuts 4
		double NumConst =       iJet->chargedMultiplicity()+iJet->neutralMultiplicity(); //cuts 5
		double MUF      =       iJet->muonEnergyFraction(); 
		double EUF      =       iJet->electronEnergyFraction();
		double CHM      =       iJet->chargedMultiplicity();
		double BJP		 =       iJet->bDiscriminator(cSV_bTag1) + iJet->bDiscriminator(cSV_bTag2);
		//APPLYING TIGHT QUALITY CUTS
		if (NHF > .9) continue;
		if (NEMF > .9) continue;
		if (NumConst <= 1) continue;
		if (MUF >= .8) continue; //MAKE SURE THE AREN'T MUONS
		if (EUF >= .8) continue; //MAKE SURE THE AREN'T ELECTRONS
		//ADDITIONAL CUTS BECAUSE OF TIGHT ETA CUT
		if (CHF == 0) continue;
		if (CHM == 0) continue;
		//if (CEMF > .99) continue;
		if (CEMF > .90)  continue;

		if(recoMuon!=0 && recoElectron!=0){
    	if(sqrt(dR2(iJet->eta(), recoMuon->eta(), iJet->phi(), recoMuon->phi())) > 0.35 && sqrt(dR2(iJet->eta(), recoElectron->eta(), iJet->phi(), recoElectron->phi())) > 0.35 ){

			if(bJet == 0 && BJP > 0.4184){
				bJet=&(*(iJet));
			}
			else if(Jet == 0){
				Jet=&(*(iJet));
			}
			else{
        		combinedJetsP4 = combinedJetsP4 + iJet->p4();
        	}
      	}
		}
	}

	m_eventsWeight->Fill(0.5, eventCount); //event count root file
	countTable(eventCount); //event count csv file, safe from corruption if the job crashes before completing

		if(genMuon!=0 && genElectron!=0 && antibGenJet!=0 && bGenJet!=0){ //pairing the gen jets with the muon or electron side of the ttbar
			if(genMuon->pdgId()<0){eJet=bGenJet; muJet=antibGenJet; }
			if(genMuon->pdgId()>0){eJet=antibGenJet; muJet=bGenJet; }

			if(muNu!=0 && eNu!=0 && recoMuon!=0 && recoElectron!=0 && tquark != 0 && antitquark != 0 && bJet != 0 && Jet != 0){	
				csvTable(genMuon,genElectron,muNu,eNu,tquark,antitquark,binNumber(genMuon),recoMuon,recoElectron,muJet,eJet,combinedGenJetsP4,bJet,Jet,combinedJetsP4,Met,eventCount,ematches);
			}
		}


}





//HELPERS
double NNstudies::dR2(double eta1, double eta2, double phi1, double phi2) {
    double deta = eta1 - eta2;
    double dphi = dPhi(phi1, phi2);
    return deta*deta + dphi*dphi;
}
double NNstudies::dPhi(double phi1, double phi2) {
    double raw_dphi = phi1 - phi2;
    if (fabs(raw_dphi) < ROOT::Math::Pi()) return raw_dphi;
    double region = std::round(raw_dphi / (2.*ROOT::Math::Pi()));
    return raw_dphi - 2.*ROOT::Math::Pi()*region;
}

//check if a lepton can be traced back to a W boson and then to a top
bool NNstudies::tWfinder(const edm::Event& iEvent, const reco::GenParticle* lepton) {

    		bool ttbar=false;
    		int iStatus;

    		const reco::Candidate* iParticle = lepton->mother();

    		while(iStatus!=4){  //status=4 is the initial proton
    			iStatus = iParticle->status();
    			
    			if(abs(iParticle->pdgId())==24){ //found W
    				while(iStatus!=4){
    				iParticle = iParticle->mother();
    				iStatus = iParticle->status();

    			   	if(abs(iParticle->pdgId())==6){ ttbar=true; 
    			   		break;
    			   	}
					}

    			}

    			iParticle = iParticle->mother();
    		}

		if(ttbar==true){return true;}
		else{return false;}
}

//check if a gen particle can be traced back to a top quark
bool NNstudies::tfinder(const edm::Event& iEvent, const reco::GenParticle* quark) {

    		bool ttbar=false;
    		int iStatus;

    		const reco::Candidate* iParticle = quark->mother();
    		iStatus = iParticle->status();

    		while(iStatus!=4){  //status=4 is the initial proton

    			   	if(abs(iParticle->pdgId())==6){ 
    			   		ttbar=true; 
    			   		break;
    			   	}
    			   iParticle = iParticle->mother();
    			   iStatus = iParticle->status();
    		}

		if(ttbar==true){return true;}
		else{return false;}
}

bool NNstudies::electronMVAcut(float pt, float eta, float bdt){ 
	if(pt < 10) return false;
	if(eta<0.8){
		if(pt < 25){
			if (bdt < (4.277 + 0.112 * (pt - 25))) return false;
		}
		else if( bdt < 4.277) return false; 
	}
	else if(eta < 1.479){
		if(pt < 25){
			if(bdt < (3.152 + 0.060 * (pt-25))) return false;
		}
		else if(bdt < 3.152) return false;
	}
	else if(eta<2.5){
		if(pt < 25){
			if(bdt < (2.359 + 0.087 * (pt-25))) return false;
		}
		else if(bdt < 2.359) return false;
	}

	return true;
}

bool NNstudies::passElectronTrig(const edm::Event& iEvent) {
  bool passTriggers = false;

  //std::cout <<"checking electron trigger paths "<<std::endl;
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(m_trigResultsToken, triggerResults);

  //std::cout <<"grabbing trigger names"<<std::endl;
  const edm::TriggerNames& trigNames = iEvent.triggerNames(*triggerResults); //SEGFAULT
  //std::cout <<"grabbing trigger results "<<std::endl;
  //std::cout <<"looping over paths to pass"<<std::endl;
  for(size_t i = 0; i < trigNames.size(); ++i) {
    const std::string &name = trigNames.triggerName(i);
    for(auto& pathName : m_electronPathsToPass){
      if((name.find(pathName) != std::string::npos )){
        if(triggerResults->accept(i)){
          passTriggers = true;
        }
      }
    }
  }

  return passTriggers;
}

//find which pt bin the reco muonfalls into
int NNstudies::binNumber(const reco::GenParticle* muon){

	double muonPT = muon->pt();

	for(int i=0; i<16; i++){
		if(muonPT >= binEdges[i] && muonPT < binEdges[i+1]){
			return i+1;
		}
	}
	if(muonPT > binEdges[16]){return 17;}
	return 0;
}

//fill the event count csv file to keep track of how many events are analyzed for scaling to data
void NNstudies::countTable( int count){
	std::ofstream myCountfile;
	myCountfile.open("count.csv",std::ios_base::app);
	myCountfile << count << "\n ";
	myCountfile.close();
}

//the csv table filler for extracting NN data
void NNstudies::csvTable(const reco::GenParticle* genMuon, const reco::GenParticle* genElectron, const reco::GenParticle* muNu, const reco::GenParticle* eNu, const reco::GenParticle* tquark, const reco::GenParticle* antitquark, int binNumber, const pat::Muon* muon, const reco::GsfElectron* electron, const reco::GenJet* muJet, const reco::GenJet* eJet, math::XYZTLorentzVector combinedGenJets, const pat::Jet* bJet, const pat::Jet* Jet, math::XYZTLorentzVector combinedJets, const pat::MET Met, double weight, int ematches) {


math::XYZTLorentzVector muonP4 = muon->p4();
math::XYZTLorentzVector electronP4 = electron->p4();
math::XYZTLorentzVector bjetP4 = bJet->p4();
math::XYZTLorentzVector jetP4=Jet->p4();
math::XYZTLorentzVector muJetP4 = muJet->p4();
math::XYZTLorentzVector eJetP4=eJet->p4();
math::XYZTLorentzVector genMuonP4 = genMuon->p4();
math::XYZTLorentzVector genElectronP4 = genElectron->p4();
math::XYZTLorentzVector muNuP4 = muNu->p4();
math::XYZTLorentzVector eNuP4 = eNu->p4();
math::XYZTLorentzVector tquarkP4 = tquark->p4();
math::XYZTLorentzVector antitquarkP4 = antitquark->p4();
math::XYZTLorentzVector metP4 = Met.p4();


std::ofstream myfile;
myfile.open(m_dataSaveFile,std::ios_base::app);
myfile << muonP4.Px() << ", "
	   << muonP4.Py() << ", "
       << muonP4.Pz() << ", "
       << muonP4.E()*abs(muon->pdgId())/muon->pdgId() << ", "
       << electronP4.Px() << ", "
       << electronP4.Py() << ", "
       << electronP4.Pz() << ", "
       << electronP4.E()*abs(electron->pdgId())/electron->pdgId()  << ", "
       << bjetP4.Px() << ", "
       << bjetP4.Py() << ", "
       << bjetP4.Pz() << ", "
       << bjetP4.E()  << ", "
       << jetP4.Px() << ", "
       << jetP4.Py() << ", "
       << jetP4.Pz() << ", "
       << jetP4.E()  << ", "
       << combinedJets.Px() <<", "
       << combinedJets.Py() <<", "
       << combinedJets.Pz() <<", "
       << combinedJets.E()  <<", "
       << muJetP4.Px() << ", "
       << muJetP4.Py() << ", "
       << muJetP4.Pz() << ", "
       << muJetP4.E()  << ", "
       << eJetP4.Px() << ", "
       << eJetP4.Py() << ", "
       << eJetP4.Pz() << ", "
       << eJetP4.E()  << ", "
       << combinedGenJets.Px() <<", "
       << combinedGenJets.Py() <<", "
       << combinedGenJets.Pz() <<", "
       << combinedGenJets.E()  <<", "
       << metP4.Px() << ", "
       << metP4.Py() <<", "
       << metP4.E() <<", "
       << weight << ", "
       << binNumber << ", "
       << genMuonP4.Px() << ", "
       << genMuonP4.Py() << ", "
       << genMuonP4.Pz() << ", "
       << genMuonP4.E()*abs(genMuon->pdgId())/genMuon->pdgId()  << ", "
       << genElectronP4.Px() << ", "
       << genElectronP4.Py() << ", "
       << genElectronP4.Pz() << ", "
       << genElectronP4.E()*abs(genElectron->pdgId())/genElectron->pdgId()  << ", "
       << muNuP4.Px() << ", "
       << muNuP4.Py() << ", "
       << muNuP4.Pz() << ", "
       << muNuP4.E()  << ", "
       << eNuP4.Px() << ", "
       << eNuP4.Py() << ", "
       << eNuP4.Pz() << ", "
       << eNuP4.E()  << ", "
       << antitquarkP4.Px() << ", "
       << antitquarkP4.Py() << ", "
       << antitquarkP4.Pz() << ", "
       << antitquarkP4.E() << ", "
       << tquarkP4.Px() << ", "
       << tquarkP4.Py() << ", "
       << tquarkP4.Pz() << ", "
       << tquarkP4.E() << ", "
       << ematches <<"\n ";


myfile.close();

}


// ------------ method called once each job just before starting event loop  ------------
void
NNstudies::beginJob() {

	std::ifstream checkmyfile;
	std::ofstream myfile;

	checkmyfile.open(m_dataSaveFile, std::ios_base::app)

	if(!checkmyfile){ //give it a header of the file doesn't exist, otherwise skip the header and append new data to the end of the file
		myfile.open(m_dataSaveFile,std::ios_base::app);
		myfile<<"muonP1,muonP2,muonP3,muonP4,electronP1,electronP2,electronP3,electronP4,bJetP1,bJetP2,bJetP3,bJetP4,JetP1,JetP2,JetP3,JetP4,combinedJetsP1,combinedJetsP2,combinedJetsP3,combinedJetsP4,muJetP1,muJetP2,muJetP3,muJetP4,eJetP1,eJetP2,eJetP3,eJetP4,combinedGenJetsP1,combinedGenJetsP2,combinedGenJetsP3,combinedGenJetsP4,METP1,METP2,METP4,eventWeight,binNumber,genMuonP1,genMuonP2,genMuonP3,genMuonP4,genElectronP1,genElectronP2,genElectronP3,genElectronP4,muNuP1,muNuP2,muNuP3,muNuP4,eNuP1,eNuP2,eNuP3,eNuP4,antitquarkP1,antitquarkP2,antitquarkP3,antitquarkP4,tquarkP1,tquarkP2,tquarkP3,tquarkP4,ematches\n";
		myfile.close();
	}

	edm::Service<TFileService> fs;

	TFileDirectory countFolder = fs->mkdir("event_count");
	
	m_histoMaker.book(fs->mkdir("Analysis"));

	m_eventsWeight = {countFolder.make<TH1D>("eventsWeight","number of events weighted", 1, 0.0, 1)};

	
}


// ------------ method called once each job just after ending the event loop  ------------
void
NNstudies::endJob() {
	
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
NNstudies::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NNstudies);
