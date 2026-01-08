#include "midi.h"
#include <fstream>
#include <math.h>

#include "../views/settings/configEditor.hpp"
#define SI_CONVERT_GENERIC
#include "SimpleIni.h"

extern ConfigEditor* config;
extern CSimpleIniA ini_gmlist;

typedef struct instrument{
	unsigned char id, type;
}instrument;

instrument* instrumentList;

int instrumentExists(unsigned char id, unsigned char type)
{

	for (unsigned i = 0; i < fm->instrumentCount; i++)
	{
		if (instrumentList[i].id == id && instrumentList[i].type == type)
			return i;
	}
	return -1;
}

int addInstrument(int id, unsigned char type)
{
	// out of range values (stupid midis!)
	if (type == 1 && (id<24 || id > 87))
	{
		id = 23;
	}
	else if (type == 0 && id > 127)
	{
		id = 0;
	}
	int i;
	if ((i = instrumentExists(id, type)) < 0)
	{
		string instrumentFile, instrumentName;
		if (type == 0)
		{
			instrumentFile = ini_gmlist.GetValue("melodic", int2str[id].c_str(), "0");
			instrumentName = midiProgramNames[id];
		}
		else
		{
			instrumentFile = ini_gmlist.GetValue("percussion", int2str[id].c_str(), "0");
			instrumentName = midiPercussionNames[id - 23];
		}
		if (mt_loadInstrument(fm, string(string(appdir + "/instruments/") + instrumentFile + string(".mdti")).c_str(), fm->instrumentCount) < 0)
		{
			mt_resizeInstrumentList(fm, fm->instrumentCount+1);
		}

		sprintf(&fm->instrument[fm->instrumentCount - 1].name[0], "%s", instrumentName.c_str());

		instrumentList = (instrument*)realloc(instrumentList, sizeof(instrument)*fm->instrumentCount);

		instrumentList[fm->instrumentCount - 1].id = id; // if unknown percussion, still store its original ID to avoid adding it multiple times
		instrumentList[fm->instrumentCount - 1].type = type;

		return fm->instrumentCount - 1;
	}
	else
	{
		return i;
	}

}

int midiImport(const char* filename)
{
	int currentVol = fm->_globalVolume;
	mt_clearSong(fm);
	mt_resizeInstrumentList(fm, 0);
	for (int i = 0; i < 128; ++i)
	{
		addInstrument(i, 0);
	}
	for (int i = 24; i < 88; ++i)
	{
		addInstrument(i, 1);
	}

	return mt_loadSong(fm, filename);
}